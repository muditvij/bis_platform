from backend.neo4j_rag import neo4j_rag_engine

print(f"Neo4j Connected: {neo4j_rag_engine.is_connected}")

query = "What Indian Standard covers motorcycle safety helmets and where can it be tested?"
print(f"\n--- Testing Search: '{query}' ---")
res = neo4j_rag_engine.search(query, top_k=2)

print(f"Hits found: {len(res['hits'])}")
for h in res["hits"]:
    entry = h["entry"]
    print(f"- [{entry.get('is_number')}] {entry.get('title')} (score: {h['score']})")
    print(f"  Schemes: {entry.get('schemes')}")
    print(f"  QCOs: {[q['title'] for q in entry.get('qcos', [])]}")
    print(f"  Clauses: {[c['clause'] + ': ' + c['title'] for c in entry.get('clauses', [])]}")

print(f"\nMulti-hop trace: {res.get('multi_hop_trace')}")
print(f"Subgraph nodes: {len(res['subgraph']['nodes'])}, edges: {len(res['subgraph']['edges'])}")

print("\n--- Testing Full Graph Fetch ---")
fg = neo4j_rag_engine.get_full_graph(limit=20)
print(f"Full graph sample: {fg['stats']}")
