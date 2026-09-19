import sqlite3
import os
import json
from typing import Optional, List, Dict, Any

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./bis_metadata.db")

def get_db(db_path: str = SQLITE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = SQLITE_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id          TEXT PRIMARY KEY,
            topic       TEXT NOT NULL,
            is_number   TEXT,
            title       TEXT NOT NULL,
            title_hi    TEXT,
            source_url  TEXT,
            doc_url     TEXT,
            portal_url  TEXT,
            verified    BOOLEAN DEFAULT 1,
            agent_ids   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            skill_id    TEXT NOT NULL REFERENCES skills(id),
            content     TEXT NOT NULL,
            keywords    TEXT,
            chunk_index INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            source_id   TEXT NOT NULL,
            target_id   TEXT NOT NULL,
            edge_type   TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            PRIMARY KEY (source_id, target_id, edge_type)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            name        TEXT PRIMARY KEY,
            label       TEXT,
            count       INTEGER DEFAULT 0
        );
        """)
        conn.commit()

def upsert_skill(skill: Dict[str, Any], db_path: str = SQLITE_PATH):
    with get_db(db_path) as conn:
        conn.execute("""
        INSERT INTO skills (id, topic, is_number, title, title_hi, source_url, doc_url, portal_url, verified, agent_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            topic=excluded.topic,
            is_number=excluded.is_number,
            title=excluded.title,
            title_hi=excluded.title_hi,
            source_url=excluded.source_url,
            doc_url=excluded.doc_url,
            portal_url=excluded.portal_url,
            verified=excluded.verified,
            agent_ids=excluded.agent_ids;
        """, (
            skill["id"],
            skill.get("topic", "standard"),
            skill.get("is_number"),
            skill["title"],
            skill.get("title_hi"),
            skill.get("source_url"),
            skill.get("doc_url"),
            skill.get("portal_url"),
            1 if skill.get("verified", True) else 0,
            json.dumps(skill.get("agent_ids", []))
        ))
        conn.commit()

def upsert_chunk(chunk: Dict[str, Any], db_path: str = SQLITE_PATH):
    with get_db(db_path) as conn:
        conn.execute("""
        INSERT INTO chunks (id, skill_id, content, keywords, chunk_index)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            skill_id=excluded.skill_id,
            content=excluded.content,
            keywords=excluded.keywords,
            chunk_index=excluded.chunk_index;
        """, (
            chunk["id"],
            chunk["skill_id"],
            chunk["content"],
            json.dumps(chunk.get("keywords", [])),
            chunk.get("chunk_index", 0)
        ))
        conn.commit()

def upsert_edge(source_id: str, target_id: str, edge_type: str, weight: float = 1.0, db_path: str = SQLITE_PATH):
    with get_db(db_path) as conn:
        conn.execute("""
        INSERT INTO edges (source_id, target_id, edge_type, weight)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
            weight=excluded.weight;
        """, (source_id, target_id, edge_type, weight))
        conn.commit()

def upsert_topic(name: str, label: str, db_path: str = SQLITE_PATH):
    with get_db(db_path) as conn:
        conn.execute("""
        INSERT INTO topics (name, label, count)
        VALUES (?, ?, 1)
        ON CONFLICT(name) DO UPDATE SET
            count = count + 1;
        """, (name, label))
        conn.commit()
