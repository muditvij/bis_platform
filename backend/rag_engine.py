import os
import json
import logging
from typing import List, Dict, Any, Optional
from backend.db import get_db, SQLITE_PATH

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, chroma_dir: str = "./chroma_data", db_path: str = SQLITE_PATH):
        self.chroma_dir = chroma_dir
        self.db_path = db_path
        self.chroma_client = None
        self.collection = None
        self.embed_model = None
        self.initialized = False
        self._init_engine()

    def _init_engine(self):
        try:
            import chromadb
            from fastembed import TextEmbedding

            os.makedirs(self.chroma_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="bis_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            # FastEmbed ONNX lightweight CPU model
            self.embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            self.initialized = True
            self._ensure_indexed()
        except Exception as e:
            logger.warning(f"ChromaDB / FastEmbed init failed: {e}. Falling back to SQLite/BM25 mode.")
            self.initialized = False

    def _ensure_indexed(self):
        if not self.initialized or not self.collection:
            return

        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"ChromaDB collection 'bis_chunks' already has {count} entries.")
                return

            logger.info("Indexing chunks into ChromaDB...")
            with get_db(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.id, c.content, c.keywords, s.id as skill_id, s.title, s.topic, s.is_number, s.agent_ids
                    FROM chunks c
                    JOIN skills s ON c.skill_id = s.id
                """)
                rows = cursor.fetchall()

            if not rows:
                return

            ids = [r["id"] for r in rows]
            docs = [r["content"] for r in rows]
            metadatas = []
            for r in rows:
                metadatas.append({
                    "skill_id": r["skill_id"],
                    "title": r["title"] or "",
                    "topic": r["topic"] or "",
                    "is_number": r["is_number"] or "",
                    "agent_ids": r["agent_ids"] or "[]"
                })

            # FastEmbed returns a generator of numpy arrays
            embeddings = list(self.embed_model.embed(docs))
            embeddings_list = [emb.tolist() for emb in embeddings]

            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            logger.info(f"Indexed {len(ids)} chunks into ChromaDB.")

            # Pre-compute similarity edges for graph (threshold >= 0.75)
            self._compute_and_save_similarity_edges(ids, embeddings_list)
        except Exception as e:
            logger.error(f"Error during ChromaDB indexing: {e}")

    def _compute_and_save_similarity_edges(self, ids: List[str], embeddings: List[List[float]], threshold: float = 0.75):
        try:
            import numpy as np
            from backend.db import upsert_edge

            mat = np.array(embeddings)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normed = mat / norms
            sim_matrix = np.dot(normed, normed.T)

            n = len(ids)
            edge_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    score = float(sim_matrix[i, j])
                    if score >= threshold:
                        upsert_edge(ids[i], ids[j], "similar", round(score, 3), self.db_path)
                        edge_count += 1
            logger.info(f"Stored {edge_count} chunk-to-chunk similarity edges in SQLite.")
        except Exception as e:
            logger.warning(f"Could not compute similarity edges: {e}")

    def search(self, query: str, top_k: int = 5, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        if self.initialized and self.collection and self.embed_model:
            try:
                q_emb = list(self.embed_model.embed([query]))[0].tolist()
                query_res = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k
                )

                if query_res and query_res["ids"] and len(query_res["ids"][0]) > 0:
                    for i in range(len(query_res["ids"][0])):
                        dist = query_res["distances"][0][i] if query_res.get("distances") else 0.5
                        # Cosine distance to similarity: sim = 1 - dist
                        similarity = max(0.0, min(1.0, 1.0 - dist))
                        meta = query_res["metadatas"][0][i]
                        
                        # Filter by agent if requested
                        if agent_id:
                            agents = json.loads(meta.get("agent_ids", "[]"))
                            if agents and agent_id not in agents:
                                continue

                        results.append({
                            "chunk_id": query_res["ids"][0][i],
                            "content": query_res["documents"][0][i],
                            "metadata": meta,
                            "score": round(similarity, 4),
                            "source": "chromadb"
                        })
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")

        return results

# Singleton instance
rag_engine = RAGEngine()
