import os
import glob
import re
import json
from typing import Dict, Any, List, Tuple
from backend.db import init_db, upsert_skill, upsert_chunk, upsert_topic, upsert_edge

def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parses simple YAML frontmatter and returns (metadata_dict, markdown_body)."""
    meta: Dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_yaml = parts[1]
            body = parts[2].strip()
            
            # Simple line-by-line parser for clean execution without strict pyyaml
            current_list_key = None
            for line in raw_yaml.splitlines():
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                
                # Check for list item
                if line_str.startswith("- "):
                    val = line_str[2:].strip().strip('"\'')
                    if current_list_key:
                        meta.setdefault(current_list_key, []).append(val)
                    continue

                if ":" in line_str:
                    k, v = line_str.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v == "" or v == "[]":
                        current_list_key = k
                        meta[k] = []
                    else:
                        current_list_key = None
                        if v.lower() == "true":
                            meta[k] = True
                        elif v.lower() == "false":
                            meta[k] = False
                        else:
                            meta[k] = v.strip('"\'')

    return meta, body

def load_and_index_skills_to_sqlite(
    skills_dir: str = "data/skills",
    db_path: str = "./bis_metadata.db"
) -> Dict[str, int]:
    init_db(db_path)
    files = glob.glob(os.path.join(skills_dir, "*.md"))
    
    skill_count = 0
    chunk_count = 0
    
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        meta, body = parse_frontmatter(content)
        skill_id = meta.get("id") or os.path.splitext(os.path.basename(filepath))[0]
        meta["id"] = skill_id

        # Upsert skill
        upsert_skill(meta, db_path)
        skill_count += 1

        # Track topic
        topic = meta.get("topic", "standard")
        upsert_topic(topic, topic.replace("_", " ").title(), db_path)

        # Build chunk content (Concatenate title + summary + keywords + next_steps)
        chunk_id = f"{skill_id}-chunk-0"
        keywords = meta.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        chunk_data = {
            "id": chunk_id,
            "skill_id": skill_id,
            "content": f"{meta.get('title', '')}\n\n{body}",
            "keywords": keywords,
            "chunk_index": 0
        }
        upsert_chunk(chunk_data, db_path)
        chunk_count += 1

        # Edge: Skill contains Chunk
        upsert_edge(skill_id, chunk_id, "contains", 1.0, db_path)

        # Edge: Chunk tagged with Topic
        upsert_edge(chunk_id, f"topic-{topic}", "tagged_with", 1.0, db_path)

    return {"skills": skill_count, "chunks": chunk_count}

if __name__ == "__main__":
    res = load_and_index_skills_to_sqlite()
    print(f"Loaded into SQLite: {res}")
