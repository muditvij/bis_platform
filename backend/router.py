import os
import json
import logging
import urllib.request
from typing import Dict, Any, List, Optional
from backend.agents_catalog import AGENTS_CATALOG

logger = logging.getLogger(__name__)

ROUTER_MODEL = os.environ.get("GROQ_MODEL_ROUTER", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_groq_api_key() -> Optional[str]:
    # Check common env names
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
    if key:
        return key.split(",")[0].strip()
    return None

def route_query_fast(query: str, selected_agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fast query classification and agent routing.
    If user already selected a marketplace agent, scopes directly to it.
    Otherwise, uses llama-3.1-8b-instant with temperature 0.0 to route.
    """
    if selected_agent_id:
        return {
            "intent": "agent_directed",
            "agent_ids": [selected_agent_id],
            "reasoning": f"User explicitly selected specialist agent '{selected_agent_id}' from marketplace.",
            "is_number_detected": None,
            "model_used": "direct"
        }

    api_key = get_groq_api_key()
    if not api_key:
        # Heuristic fallback if no API key configured
        return _heuristic_route(query)

    agents_summary = "\n".join([f"- {a['id']}: {a['name']} ({', '.join(a.get('applicable_standards', []))})" for a in AGENTS_CATALOG[:12]])
    prompt = f"""You are a query router for BIS Sahayak (Bureau of Indian Standards compliance assistant).
Classify the user query and pick 1 or 2 most relevant agent IDs.

Specialist Agents:
{agents_summary}

Respond ONLY with valid JSON with this exact structure:
{{
  "intent": "standard_lookup" | "scheme_guidance" | "consumer_complaint" | "hallmarking" | "general_bis" | "out_of_scope",
  "agent_ids": ["agent-id-1"],
  "reasoning": "1 sentence explanation",
  "is_number_detected": "IS XXXX" or null
}}

User Query: {query}"""

    payload = {
        "model": ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise JSON query classifier. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "BIS-Sahayak-Router/1.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["model_used"] = ROUTER_MODEL
            return parsed
    except Exception as e:
        logger.warning(f"Router LLM call failed ({e}). Using heuristic classification.")
        return _heuristic_route(query)

def _heuristic_route(query: str) -> Dict[str, Any]:
    q = query.lower()
    intent = "standard_lookup"
    agent_ids = ["general-compliance"]
    reasoning = "Heuristic rule matching on query keywords."

    if any(w in q for w in ["gold", "silver", "hallmark", "huid", "jewel"]):
        intent = "hallmarking"
        agent_ids = ["hallmarking-expert"]
    elif any(w in q for w in ["helmet", "4151", "two wheeler", "bike"]):
        intent = "standard_lookup"
        agent_ids = ["auto-parts"]
    elif any(w in q for w in ["fire", "extinguisher", "15683"]):
        intent = "standard_lookup"
        agent_ids = ["fire-safety"]
    elif any(w in q for w in ["footwear", "shoe", "chappal", "leather"]):
        intent = "standard_lookup"
        agent_ids = ["footwear-leather"]
    elif any(w in q for w in ["scheme", "isi", "crs", "fmcs", "license", "licence"]):
        intent = "scheme_guidance"
        agent_ids = ["isi-mark"]
    elif any(w in q for w in ["complaint", "fake", "fraud", "care app", "report"]):
        intent = "consumer_complaint"
        agent_ids = ["consumer-advocate"]

    return {
        "intent": intent,
        "agent_ids": agent_ids,
        "reasoning": reasoning,
        "is_number_detected": None,
        "model_used": "heuristic-fallback"
    }
