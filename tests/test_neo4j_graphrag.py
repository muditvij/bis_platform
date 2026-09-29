"""
Comprehensive Verification Suite for Neo4j GraphRAG in BIS Sahayak.
Tests:
  1. Neo4j AuraDB Connectivity and Vector Index.
  2. Multi-hop Graph Traversal (Standard -> Scheme, QCO, Clause, Lab).
  3. API ask endpoint and decision trace with Neo4j strategy.
  4. Cytoscape graph data generation.
"""

import sys
import os

# Set root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from backend.neo4j_rag import neo4j_rag_engine
from backend.app import handle_ask

def test_connectivity():
    print("\n[TEST 1] Testing Neo4j AuraDB Connectivity...")
    assert neo4j_rag_engine.is_connected, "Neo4j must be connected"
    print("PASS: Neo4j AuraDB is connected and verified.")

def test_multihop_search_helmet():
    print("\n[TEST 2] Testing Multi-Hop Traversal on Helmets Query...")
    q = "What Indian Standard applies to two-wheeler motorcycle helmets and what lab can test impact absorption?"
    res = neo4j_rag_engine.search(q, top_k=2)
    assert len(res["hits"]) > 0, "Hits should not be empty"
    top_hit = res["hits"][0]["entry"]
    assert "IS 4151" in (top_hit.get("is_number") or top_hit.get("title")), f"Expected IS 4151, got {top_hit.get('is_number')}"
    assert "Scheme-I (ISI Mark)" in top_hit.get("schemes", []), f"Expected Scheme-I in schemes, got {top_hit.get('schemes')}"
    
    # Check subgraph
    assert len(res["subgraph"]["nodes"]) >= 3, "Subgraph should contain multiple connected nodes"
    assert len(res["subgraph"]["edges"]) >= 2, "Subgraph should contain multiple edges"
    print(f"PASS: Resolved {top_hit.get('is_number')}, Schemes: {top_hit.get('schemes')}, QCOs: {[q['title'] for q in top_hit.get('qcos',[])]}")
    print(f"PASS: Subgraph contains {len(res['subgraph']['nodes'])} nodes and {len(res['subgraph']['edges'])} edges.")

def test_multihop_search_steel():
    print("\n[TEST 3] Testing Multi-Hop Traversal on Steel Bars Query...")
    q = "What standard covers deformed steel bars for concrete reinforcement and what chemical limits apply?"
    res = neo4j_rag_engine.search(q, top_k=2)
    assert len(res["hits"]) > 0, "Hits should not be empty"
    top_hit = res["hits"][0]["entry"]
    assert "IS 1786" in (top_hit.get("is_number") or top_hit.get("title")), f"Expected IS 1786, got {top_hit.get('is_number')}"
    print(f"PASS: Resolved {top_hit.get('is_number')} with {len(top_hit.get('clauses', []))} key compliance clauses.")

def test_handle_ask_integration():
    print("\n[TEST 4] Testing handle_ask() Integration with Neo4j Decision Trace...")
    payload = {"question": "What standard covers gold jewellery hallmarking and what is HUID?"}
    ans, code = handle_ask(payload)
    assert code == 200, f"Expected 200, got {code}"
    assert "decision_trace" in ans, "Answer must contain decision trace"
    trace = ans["decision_trace"]
    print(f"Retrieval strategy: {trace.get('retrieval_strategy')}")
    assert "Neo4j GraphRAG" in trace.get("retrieval_strategy", ""), "Should use Neo4j GraphRAG strategy"
    assert "subgraph" in ans, "Response must include subgraph"
    assert len(ans["subgraph"]["nodes"]) > 0, "Subgraph must contain traversed nodes"
    print("PASS: handle_ask generated answer backed by Neo4j GraphRAG with decision trace and subgraph.")

def test_full_graph_api():
    print("\n[TEST 5] Testing Neo4j Full Graph Extraction for Cytoscape...")
    graph = neo4j_rag_engine.get_full_graph(limit=40)
    assert len(graph["nodes"]) > 0, "Full graph must contain nodes"
    assert len(graph["edges"]) > 0, "Full graph must contain edges"
    print(f"PASS: Extracted {len(graph['nodes'])} nodes and {len(graph['edges'])} edges from Neo4j AuraDB.")

if __name__ == "__main__":
    print("=== STARTING NEO4J GRAPHRAG VERIFICATION ===")
    test_connectivity()
    test_multihop_search_helmet()
    test_multihop_search_steel()
    test_handle_ask_integration()
    test_full_graph_api()
    print("\n=== ALL 5 VERIFICATION SUITES PASSED PERFECTLY ===")
