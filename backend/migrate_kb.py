import json
import os
import re
from typing import List, Dict, Any

def sanitize_filename(name: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_\-]', '_', name.lower())
    return re.sub(r'_+', '_', s).strip('_')

def map_entry_to_agents(entry: Dict[str, Any]) -> List[str]:
    topic = entry.get("topic", "").lower()
    is_num = entry.get("is_number", "")
    title = (entry.get("title", "") + " " + " ".join(entry.get("keywords", []))).lower()

    agent_ids = set()

    if "helmet" in title or "4151" in is_num or "tyre" in title or "battery" in title:
        agent_ids.add("auto-parts")
    if "fire" in title or "15683" in is_num:
        agent_ids.add("fire-safety")
    if "gold" in title or "silver" in title or "hallmark" in title or "huid" in title or "1417" in is_num or "2112" in is_num:
        agent_ids.add("hallmarking-expert")
    if "footwear" in title or "leather" in title or "shoe" in title or "15844" in is_num or "6721" in is_num:
        agent_ids.add("footwear-leather")
    if "toy" in title or "9873" in is_num:
        agent_ids.add("toys-children")
    if "wire" in title or "cable" in title or "694" in is_num:
        agent_ids.add("cables-conductors")
    if "cement" in title or "steel" in title or "1786" in is_num or "1489" in is_num:
        agent_ids.add("steel-cement")
    if "solar" in title or "pv" in title or "14286" in is_num:
        agent_ids.add("solar-renewables")
    if "water" in title or "14543" in is_num or "13428" in is_num:
        agent_ids.add("packaged-water")
    if "food" in title or "milk" in title or "oil" in title:
        agent_ids.add("food-agro")
    if "medical" in title or "mask" in title or "16289" in is_num:
        agent_ids.add("medical-devices")
    if "electronic" in title or "mobile" in title or "13252" in is_num or "crs" in topic or "scheme-ii" in title:
        agent_ids.add("electronics-it")
    if "scheme-i" in title or "isi" in title:
        agent_ids.add("isi-mark")
    if "fmcs" in title or "foreign" in title:
        agent_ids.add("fmcs-advisor")
    if "complaint" in title or "care app" in title or "fake" in title or "consumer" in title:
        agent_ids.add("consumer-advocate")

    if not agent_ids:
        if topic == "standard":
            agent_ids.add("general-compliance")
        elif topic == "scheme":
            agent_ids.add("isi-mark")
        else:
            agent_ids.add("general-compliance")

    return sorted(list(agent_ids))

def migrate(kb_path: str = "data/knowledge_base.json", output_dir: str = "data/skills"):
    if not os.path.exists(kb_path):
        print(f"Error: {kb_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data

    count = 0
    for item in entries:
        item_id = item.get("id", f"ENTRY-{count+1:03d}")
        agent_ids = map_entry_to_agents(item)
        keywords = item.get("keywords", [])
        
        # Build YAML frontmatter manually to avoid strict pyyaml dependency in stdlib mode
        yaml_lines = [
            "---",
            f'id: "{item_id}"',
            f'topic: "{item.get("topic", "standard")}"',
            f'is_number: "{item.get("is_number") or ""}"',
            f'title: "{item.get("title", "").replace(chr(34), "")}"',
            f'title_hi: "{item.get("title_hi", "").replace(chr(34), "")}"',
            'keywords:'
        ]
        for kw in keywords:
            clean_kw = str(kw).replace('"', '')
            yaml_lines.append(f'  - "{clean_kw}"')
        
        yaml_lines.append('agent_ids:')
        for aid in agent_ids:
            yaml_lines.append(f'  - "{aid}"')

        yaml_lines.extend([
            f'source_title: "{item.get("source_title", "").replace(chr(34), "")}"',
            f'source_url: "{item.get("source_url", "")}"',
            f'doc_url: "{item.get("doc_url", "")}"',
            f'portal_url: "{item.get("portal_url", "")}"',
            f'verified: {str(item.get("verified", True)).lower()}',
            "---",
            ""
        ])

        # Markdown body
        body_lines = [
            f'# {item.get("title", "")}',
            "",
            item.get("summary", ""),
            ""
        ]

        if item.get("summary_hi"):
            body_lines.extend([
                "## सारांश (Summary Hindi)",
                item.get("summary_hi", ""),
                ""
            ])

        next_steps = item.get("next_steps", [])
        if next_steps:
            body_lines.append("## Next Steps")
            for idx, step in enumerate(next_steps, 1):
                body_lines.append(f"{idx}. {step}")
            body_lines.append("")

        next_steps_hi = item.get("next_steps_hi", [])
        if next_steps_hi:
            body_lines.append("## अगले कदम (Next Steps Hindi)")
            for idx, step in enumerate(next_steps_hi, 1):
                body_lines.append(f"{idx}. {step}")
            body_lines.append("")

        filename = f"{sanitize_filename(item_id)}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as out:
            out.write("\n".join(yaml_lines) + "\n".join(body_lines))
        count += 1

    print(f"Successfully migrated {count} entries into skill files at {output_dir}/")

if __name__ == "__main__":
    migrate()
