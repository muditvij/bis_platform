import sqlite3
import os
import json
from typing import Dict, Any, List, Optional
from collections import deque
from backend.db import get_db, SQLITE_PATH

def build_graph_data(
    db_path: str = SQLITE_PATH,
    filter_type: Optional[str] = None,
    search_query: Optional[str] = None,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    nodes = []
    edges = []
    search_term = (search_query or "").strip().lower()

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # 1. Fetch skills
        cursor.execute("SELECT id, topic, is_number, title, title_hi, agent_ids, verified FROM skills")
        skill_rows = cursor.fetchall()

        # 2. Fetch chunks
        cursor.execute("SELECT id, skill_id, content, keywords FROM chunks")
        chunk_rows = cursor.fetchall()

        # 3. Fetch topics
        cursor.execute("SELECT name, label, count FROM topics")
        topic_rows = cursor.fetchall()

        # 4. Fetch edges
        cursor.execute("SELECT source_id, target_id, edge_type, weight FROM edges")
        edge_rows = cursor.fetchall()

    # Process skills
    if not filter_type or filter_type == "skill":
        for row in skill_rows:
            s_id = row["id"]
            title = row["title"] or ""
            is_num = row["is_number"] or ""
            topic = row["topic"] or ""
            agents = json.loads(row["agent_ids"]) if row["agent_ids"] else []

            if agent_id and agent_id not in agents:
                continue

            matches_search = bool(
                search_term and (
                    search_term in title.lower() or
                    search_term in is_num.lower() or
                    search_term in topic.lower() or
                    search_term in s_id.lower()
                )
            )

            nodes.append({
                "data": {
                    "id": s_id,
                    "label": f"{is_num} - {title}" if is_num else title[:30],
                    "type": "skill",
                    "is_number": is_num,
                    "topic": topic,
                    "agent_ids": agents,
                    "verified": bool(row["verified"]),
                    "match": matches_search
                }
            })

    # Process chunks
    if not filter_type or filter_type == "chunk":
        for row in chunk_rows:
            c_id = row["id"]
            content = row["content"] or ""
            kw = json.loads(row["keywords"]) if row["keywords"] else []

            matches_search = bool(
                search_term and (
                    search_term in content.lower() or
                    any(search_term in k.lower() for k in kw)
                )
            )

            nodes.append({
                "data": {
                    "id": c_id,
                    "label": content[:35].replace("\n", " ") + "...",
                    "type": "chunk",
                    "skill_id": row["skill_id"],
                    "match": matches_search
                }
            })

    # Process topics
    if not filter_type or filter_type == "topic":
        for row in topic_rows:
            t_name = row["name"]
            t_id = f"topic-{t_name}"
            label = row["label"] or t_name.title()
            matches_search = bool(search_term and (search_term in label.lower() or search_term in t_name.lower()))

            nodes.append({
                "data": {
                    "id": t_id,
                    "label": label,
                    "type": "topic",
                    "count": row["count"],
                    "match": matches_search
                }
            })

    valid_node_ids = {n["data"]["id"] for n in nodes}

    for row in edge_rows:
        src = row["source_id"]
        tgt = row["target_id"]
        if src in valid_node_ids and tgt in valid_node_ids:
            edges.append({
                "data": {
                    "id": f"e-{src}-{tgt}-{row['edge_type']}",
                    "source": src,
                    "target": tgt,
                    "type": row["edge_type"],
                    "weight": row["weight"]
                }
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {
                "skill": len([n for n in nodes if n["data"]["type"] == "skill"]),
                "chunk": len([n for n in nodes if n["data"]["type"] == "chunk"]),
                "topic": len([n for n in nodes if n["data"]["type"] == "topic"])
            }
        }
    }

def find_shortest_path(source_id: str, target_id: str, db_path: str = SQLITE_PATH) -> Dict[str, Any]:
    """Breadth-first search for shortest undirected path between two nodes."""
    adj: Dict[str, List[str]] = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT source_id, target_id FROM edges")
        for row in cursor.fetchall():
            u, v = row["source_id"], row["target_id"]
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

    if source_id not in adj or target_id not in adj:
        return {"path": [], "hops": 0, "found": False}

    queue = deque([[source_id]])
    visited = {source_id}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == target_id:
            return {"path": path, "hops": len(path) - 1, "found": True}

        for neighbor in adj.get(node, []):
            if neighbor not in visited and len(path) < 8:  # Cap at 8 hops
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return {"path": [], "hops": 0, "found": False}
