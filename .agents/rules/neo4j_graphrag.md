# Neo4j GraphRAG Architecture Rules & Guidelines

## Overview
BIS Sahayak utilizes a **Dual Hybrid GraphRAG Architecture** running on Neo4j AuraDB.
The pipeline combines:
1. FastEmbed ONNX (`BAAI/bge-small-en-v1.5`, 384 dimensions) vector similarity on `Chunk(embedding)`.
2. Full-Text lexical search on `Standard(is_number, title, keywords_str)`.
3. Multi-Hop Graph Traversal across:
   - `(:Product)-[:APPLIES_STANDARD]->(:Standard)`
   - `(:Standard)-[:CERTIFIED_UNDER]->(:Scheme)`
   - `(:Standard)-[:MANDATED_BY]->(:QCO)`
   - `(:Standard)-[:HAS_CLAUSE]->(:Clause)-[:TESTED_AT]->(:Laboratory)`
   - `(:Standard)-[:REFERENCES_NORMATIVE]->(:Standard)`

## Ingestion & Initialization
- Script: `backend/init_neo4j.py`
- Re-run this script whenever `data/knowledge_base.json` is updated or new lab/QCO nodes are added.

## Testing & Verification
- Test Suite: `python tests/test_neo4j_graphrag.py`
- Always verify all 5 test suites pass before deploying or pushing changes.

## Graceful Fallback
- If Neo4j AuraDB is unreachable or credentials are unconfigured, `backend/app.py` automatically falls back to:
  1. Local ChromaDB dense vector search (`backend/rag_engine.py`)
  2. Local BM25 lexical inverted index (`backend/retriever.py`)
- This ensures zero downtime during demos or offline environments.
