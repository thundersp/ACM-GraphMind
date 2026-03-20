"""
Vector Retrieval Service - lightweight semantic retrieval over user facts/messages.

This implementation keeps dependencies minimal:
- Uses deterministic hashed embeddings (no external model download required)
- Retrieves user text memories from Neo4j
- Computes cosine similarity in-process
"""

from typing import Dict, List, Any, Tuple
import math
import re
import logging
import hashlib
from neo4j import GraphDatabase
from config.settings import Settings

logger = logging.getLogger(__name__)


class VectorRetrieval:
    """Provides vector-style retrieval for unstructured text memories."""

    EMBEDDING_DIM = 256
    MAX_CANDIDATES = 500
    SNIPPET_MAX_LENGTH = 220

    def __init__(self):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                Settings.NEO4J_URI,
                auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            logger.warning("Could not connect to Neo4j for vector retrieval: %s", e)
            self.driver = None

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 8
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieve semantically similar text memories for a query.

        Returns:
            Tuple[List[context_items], retrieval_time_ms]
        """
        import time
        start = time.time()

        if not self.driver or not query.strip():
            return [], 0.0

        try:
            with self.driver.session() as session:
                records = session.run(
                    """
                    MATCH (f:Fact {user_id: $user_id})
                    WHERE f.text IS NOT NULL AND trim(f.text) <> ""
                    RETURN f.id AS id, f.text AS text, coalesce(f.confidence, 0.5) AS confidence
                    ORDER BY coalesce(f.updated_at, f.created_at, f.timestamp) DESC
                    LIMIT $limit
                    """,
                    user_id=user_id,
                    limit=self.MAX_CANDIDATES,
                )
                candidates = [dict(r) for r in records]

            if not candidates:
                return [], (time.time() - start) * 1000

            query_vec = self._embed_text(query)
            ranked: List[Dict[str, Any]] = []

            for item in candidates:
                text = item.get("text") or ""
                score = self._cosine_similarity(query_vec, self._embed_text(text))
                if score == 0.0:
                    continue

                ranked.append(
                    {
                        "type": "TextMemory",
                        "text": text,
                        "retrieval_score": round(score, 4),
                        "snippet": text[:self.SNIPPET_MAX_LENGTH],
                        "properties": {
                            "id": item.get("id"),
                            "text": text,
                            "confidence": item.get("confidence", 0.5),
                        },
                        "retrieval_source": "vector",
                    }
                )

            ranked.sort(key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
            return ranked[:top_k], (time.time() - start) * 1000

        except Exception as e:
            logger.exception("Error during vector retrieval: %s", e)
            return [], (time.time() - start) * 1000

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Lowercase tokenization for embedding."""
        return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

    @classmethod
    def _embed_text(cls, text: str) -> List[float]:
        """
        Deterministic hashed embedding vector.
        Keeps memory/query text comparable without external dependencies.
        """
        vec = [0.0] * cls.EMBEDDING_DIM
        tokens = cls._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            idx = cls._stable_index(token)
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity for normalized vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    @classmethod
    def _stable_index(cls, token: str) -> int:
        """Map token to deterministic embedding index."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big")
        return value % cls.EMBEDDING_DIM

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
