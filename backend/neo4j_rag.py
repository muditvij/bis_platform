"""
Neo4j GraphRAG Engine for BIS Sahayak.
Combines Vector Embeddings + Full-Text Search + Multi-Hop Graph Traversal on Neo4j AuraDB.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

class Neo4jGraphRAG:
    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = NEO4J_DATABASE
        self.driver = None
        self.embed_model = None
        self.is_connected = False
        self._init_engine()

    def _init_engine(self):
        try:
            logger.info("Connecting Neo4j GraphRAG driver to %s...", self.uri)
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self.embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            self.is_connected = True
            logger.info("Neo4j GraphRAG driver initialized and connected successfully.")
        except Exception as e:
            logger.warning(f"Neo4j connection error ({e}). Will fall back to ChromaDB/BM25 if needed.")
            self.is_connected = False

    def search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Hybrid Vector + Graph Multi-Hop Traversal Search.
        Returns:
            - hits: list of enriched standard/clause entries formatted for answer composer
            - subgraph: {nodes, edges} representing the traversed graph for Cytoscape.js
            - multi_hop_trace: explanation of relationships traversed
        """
        if not self.is_connected or not self.driver or not self.embed_model:
            return {"hits": [], "subgraph": {"nodes": [], "edges": []}, "multi_hop_trace": []}

        try:
            # 1. Embed query
            q_emb = list(self.embed_model.embed([query]))[0].tolist()

            # 2. Cypher Vector Search + Multi-Hop Graph Traversal
            vector_cypher = """
            CALL db.index.vector.queryNodes('chunk_vector_index', $top_k, $embedding)
            YIELD node AS chunkNode, score
            MATCH (s:Standard)-[:HAS_CHUNK]->(chunkNode)
            OPTIONAL MATCH (s)-[:CERTIFIED_UNDER]->(sc:Scheme)
            OPTIONAL MATCH (s)-[:MANDATED_BY]->(q:QCO)
            OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(cl:Clause)
            OPTIONAL MATCH (cl)-[:TESTED_AT]->(lab:Laboratory)
            OPTIONAL MATCH (s)-[:REFERENCES_NORMATIVE]->(ref:Standard)
            RETURN s.id AS id,
                   s.is_number AS is_number,
                   s.title AS title,
                   s.title_hi AS title_hi,
                   s.topic AS topic,
                   s.summary AS summary,
                   s.summary_hi AS summary_hi,
                   s.source_title AS source_title,
                   s.source_url AS source_url,
                   s.verified AS verified,
                   s.next_steps AS next_steps,
                   score,
                   collect(DISTINCT sc.name) AS schemes,
                   collect(DISTINCT {id: q.id, title: q.title, ministry: q.ministry, status: q.status, date: q.effective_date}) AS qcos,
                   collect(DISTINCT {id: cl.id, clause: cl.clause_number, title: cl.title, desc: cl.description, lab: lab.name, state: lab.state}) AS clauses,
                   collect(DISTINCT ref.is_number) AS normative_refs
            ORDER BY score DESC
            LIMIT $top_k
            """

            hits = []
            subgraph_nodes = []
            subgraph_edges = []
            seen_nodes = set()
            seen_edges = set()
            multi_hop_trace = []

            with self.driver.session() as session:
                result = session.run(vector_cypher, top_k=top_k, embedding=q_emb)
                records = list(result)

                # Fallback to full-text or direct match if vector score is low
                if not records or records[0]["score"] < 0.45:
                    clean_kw = "".join([c if c.isalnum() else " " for c in query]).strip()
                    if clean_kw:
                        ft_cypher = """
                        MATCH (s:Standard)
                        WHERE s.is_number =~ ('(?i).*' + $q + '.*')
                           OR s.title =~ ('(?i).*' + $q + '.*')
                           OR s.keywords_str =~ ('(?i).*' + $q + '.*')
                        OPTIONAL MATCH (s)-[:CERTIFIED_UNDER]->(sc:Scheme)
                        OPTIONAL MATCH (s)-[:MANDATED_BY]->(q:QCO)
                        OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(cl:Clause)
                        OPTIONAL MATCH (cl)-[:TESTED_AT]->(lab:Laboratory)
                        OPTIONAL MATCH (s)-[:REFERENCES_NORMATIVE]->(ref:Standard)
                        RETURN s.id AS id,
                               s.is_number AS is_number,
                               s.title AS title,
                               s.title_hi AS title_hi,
                               s.topic AS topic,
                               s.summary AS summary,
                               s.summary_hi AS summary_hi,
                               s.source_title AS source_title,
                               s.source_url AS source_url,
                               s.verified AS verified,
                               s.next_steps AS next_steps,
                               0.85 AS score,
                               collect(DISTINCT sc.name) AS schemes,
                               collect(DISTINCT {id: q.id, title: q.title, ministry: q.ministry, status: q.status, date: q.effective_date}) AS qcos,
                               collect(DISTINCT {id: cl.id, clause: cl.clause_number, title: cl.title, desc: cl.description, lab: lab.name, state: lab.state}) AS clauses,
                               collect(DISTINCT ref.is_number) AS normative_refs
                        LIMIT $top_k
                        """
                        ft_res = session.run(ft_cypher, q=clean_kw, top_k=top_k)
                        records.extend(list(ft_res))

                for r in records:
                    s_id = r["id"]
                    is_num = r["is_number"] or ""
                    title = r["title"] or ""
                    summary = r["summary"] or ""
                    score = float(r["score"] or 0.8)

                    # Enrich summary with Graph relational intelligence
                    enrichments = []
                    schemes = [s for s in r["schemes"] if s]
                    if schemes:
                        enrichments.append(f"Certification Scheme: {', '.join(schemes)}.")
                        for sc in schemes:
                            multi_hop_trace.append(f"{is_num or title} -> Certified under Scheme: {sc}")

                    qcos = [q for q in r["qcos"] if q.get("title")]
                    if qcos:
                        q_str = "; ".join([f"{q['title']} ({q.get('status', 'Mandatory')} under {q.get('ministry', 'Line Ministry')})" for q in qcos])
                        enrichments.append(f"Regulatory QCO: {q_str}.")
                        for q in qcos:
                            multi_hop_trace.append(f"{is_num or title} -> Mandated by QCO: {q['title']}")

                    clauses = [c for c in r["clauses"] if c.get("clause")]
                    if clauses:
                        cl_str = "; ".join([f"{c['clause']} ({c.get('title')})" + (f" [Test Lab: {c['lab']}, {c.get('state')}]" if c.get('lab') else "") for c in clauses[:2]])
                        enrichments.append(f"Key Test Clauses & Labs: {cl_str}.")
                        for c in clauses[:2]:
                            if c.get("lab"):
                                multi_hop_trace.append(f"{is_num} -> Clause: {c['clause']} -> Tested at: {c['lab']}")

                    norm_refs = [ref for ref in r["normative_refs"] if ref]
                    if norm_refs:
                        enrichments.append(f"Normative References: {', '.join(norm_refs[:3])}.")

                    enriched_summary = summary
                    if enrichments:
                        enriched_summary += "\n\n**Compliance & Regulatory Intelligence (via Neo4j Knowledge Graph):**\n- " + "\n- ".join(enrichments)

                    # Standard entry formatted for answer.py
                    entry_dict = {
                        "id": s_id,
                        "is_number": is_num,
                        "title": title,
                        "title_hi": r.get("title_hi"),
                        "topic": r.get("topic") or "standard",
                        "summary": enriched_summary,
                        "summary_hi": r.get("summary_hi"),
                        "source_title": r.get("source_title") or "BIS Indian Standard",
                        "source_url": r.get("source_url") or "https://www.services.bis.gov.in/",
                        "verified": bool(r.get("verified", True)),
                        "next_steps": r.get("next_steps") or [],
                        "schemes": schemes,
                        "qcos": qcos,
                        "clauses": clauses,
                        "normative_refs": norm_refs
                    }
                    hits.append({"score": round(score, 3), "entry": entry_dict})

                    # Build Cytoscape Subgraph elements
                    if s_id not in seen_nodes:
                        seen_nodes.add(s_id)
                        subgraph_nodes.append({
                            "data": {
                                "id": s_id,
                                "label": f"{is_num}\n{title[:25]}..." if is_num else title[:30],
                                "type": "standard",
                                "is_number": is_num
                            }
                        })

                    for sc_name in schemes:
                        sc_node_id = f"scheme-{sc_name.replace(' ', '-').lower()}"
                        if sc_node_id not in seen_nodes:
                            seen_nodes.add(sc_node_id)
                            subgraph_nodes.append({"data": {"id": sc_node_id, "label": sc_name, "type": "scheme"}})
                        e_id = f"e-{s_id}-{sc_node_id}"
                        if e_id not in seen_edges:
                            seen_edges.add(e_id)
                            subgraph_edges.append({"data": {"id": e_id, "source": s_id, "target": sc_node_id, "label": "CERTIFIED_UNDER", "type": "CERTIFIED_UNDER"}})

                    for q in qcos:
                        q_node_id = q.get("id") or f"qco-{q['title'][:15]}"
                        if q_node_id not in seen_nodes:
                            seen_nodes.add(q_node_id)
                            subgraph_nodes.append({"data": {"id": q_node_id, "label": q["title"][:28] + "...", "type": "qco", "status": q.get("status")}})
                        e_id = f"e-{s_id}-{q_node_id}"
                        if e_id not in seen_edges:
                            seen_edges.add(e_id)
                            subgraph_edges.append({"data": {"id": e_id, "source": s_id, "target": q_node_id, "label": "MANDATED_BY", "type": "MANDATED_BY"}})

                    for c in clauses[:2]:
                        c_node_id = c.get("id") or f"cl-{c['clause']}"
                        if c_node_id not in seen_nodes:
                            seen_nodes.add(c_node_id)
                            subgraph_nodes.append({"data": {"id": c_node_id, "label": f"{c['clause']}: {c['title'][:20]}", "type": "clause"}})
                        e_id = f"e-{s_id}-{c_node_id}"
                        if e_id not in seen_edges:
                            seen_edges.add(e_id)
                            subgraph_edges.append({"data": {"id": e_id, "source": s_id, "target": c_node_id, "label": "HAS_CLAUSE", "type": "HAS_CLAUSE"}})

                        if c.get("lab"):
                            lab_id = f"lab-{c['lab'].replace(' ', '-').lower()}"
                            if lab_id not in seen_nodes:
                                seen_nodes.add(lab_id)
                                subgraph_nodes.append({"data": {"id": lab_id, "label": c["lab"][:25], "type": "lab", "state": c.get("state")}})
                            e_lab_id = f"e-{c_node_id}-{lab_id}"
                            if e_lab_id not in seen_edges:
                                seen_edges.add(e_lab_id)
                                subgraph_edges.append({"data": {"id": e_lab_id, "source": c_node_id, "target": lab_id, "label": "TESTED_AT", "type": "TESTED_AT"}})

            return {
                "hits": hits,
                "subgraph": {"nodes": subgraph_nodes, "edges": subgraph_edges},
                "multi_hop_trace": multi_hop_trace
            }
        except Exception as e:
            logger.error(f"Error executing Neo4j GraphRAG search: {e}", exc_info=True)
            return {"hits": [], "subgraph": {"nodes": [], "edges": []}, "multi_hop_trace": []}

    def get_full_graph(self, filter_type: Optional[str] = None, search_query: Optional[str] = None, limit: int = 160) -> Dict[str, Any]:
        """Fetches the graph directly from Neo4j AuraDB for Cytoscape.js visualization."""
        if not self.is_connected or not self.driver:
            return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0}}

        nodes = []
        edges = []
        seen_nodes = set()
        seen_edges = set()

        search_term = (search_query or "").strip().lower()

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE NOT n:Chunk AND NOT m:Chunk
        RETURN n, labels(n)[0] AS label_n, r, type(r) AS rel_type, m, labels(m)[0] AS label_m
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher, limit=limit)
                for record in result:
                    n = record["n"]
                    lbl_n = (record["label_n"] or "standard").lower()
                    n_id = n.get("id") or str(n.id)

                    matches_search = bool(
                        search_term and (
                            search_term in str(n.get("title", "")).lower() or
                            search_term in str(n.get("name", "")).lower() or
                            search_term in str(n.get("is_number", "")).lower()
                        )
                    )

                    if n_id not in seen_nodes:
                        seen_nodes.add(n_id)
                        label_text = n.get("is_number") or n.get("title") or n.get("name") or n_id
                        if len(label_text) > 28:
                            label_text = label_text[:28] + "..."
                        nodes.append({
                            "data": {
                                "id": n_id,
                                "label": label_text,
                                "type": lbl_n,
                                "is_number": n.get("is_number", ""),
                                "verified": n.get("verified", True),
                                "match": matches_search
                            }
                        })

                    m = record["m"]
                    if m is not None:
                        lbl_m = (record["label_m"] or "node").lower()
                        m_id = m.get("id") or str(m.id)
                        if m_id not in seen_nodes:
                            seen_nodes.add(m_id)
                            m_label = m.get("is_number") or m.get("title") or m.get("name") or m_id
                            if len(m_label) > 28:
                                m_label = m_label[:28] + "..."
                            nodes.append({
                                "data": {
                                    "id": m_id,
                                    "label": m_label,
                                    "type": lbl_m,
                                    "is_number": m.get("is_number", ""),
                                    "match": bool(search_term and search_term in m_label.lower())
                                }
                            })

                        r = record["r"]
                        if r is not None:
                            edge_id = f"e-{n_id}-{m_id}-{record['rel_type']}"
                            if edge_id not in seen_edges:
                                seen_edges.add(edge_id)
                                edges.append({
                                    "data": {
                                        "id": edge_id,
                                        "source": n_id,
                                        "target": m_id,
                                        "type": record["rel_type"],
                                        "label": record["rel_type"]
                                    }
                                })

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "source": "Neo4j AuraDB"
                }
            }
        except Exception as e:
            logger.error(f"Error reading Neo4j graph for Cytoscape: {e}")
            return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0}}

    def find_shortest_path(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Finds shortest path between two nodes in Neo4j."""
        if not self.is_connected or not self.driver:
            return {"path": [], "length": 0}

        cypher = """
        MATCH (a {id: $source}), (b {id: $target})
        MATCH p = shortestPath((a)-[*]-(b))
        RETURN [node in nodes(p) | node.id] AS path_nodes, length(p) AS path_len
        """
        try:
            with self.driver.session() as session:
                res = session.run(cypher, source=source_id, target=target_id).single()
                if res:
                    return {"path": res["path_nodes"], "length": res["path_len"]}
        except Exception as e:
            logger.warning(f"Error computing shortest path: {e}")
        return {"path": [], "length": 0}

neo4j_rag_engine = Neo4jGraphRAG()
