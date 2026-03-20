import sys
import types
import unittest

# Lightweight stubs so tests can run without optional external dependencies installed.
if "neo4j" not in sys.modules:
    neo4j_stub = types.ModuleType("neo4j")

    class _GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            raise RuntimeError("neo4j not available in test env")

    neo4j_stub.GraphDatabase = _GraphDatabase
    sys.modules["neo4j"] = neo4j_stub

if "neo4j.time" not in sys.modules:
    neo4j_time_stub = types.ModuleType("neo4j.time")

    class DateTime:
        pass

    neo4j_time_stub.DateTime = DateTime
    sys.modules["neo4j.time"] = neo4j_time_stub

if "google.generativeai" not in sys.modules:
    if "google" not in sys.modules:
        google_stub = types.ModuleType("google")
        sys.modules["google"] = google_stub
    genai_stub = types.ModuleType("google.generativeai")

    def configure(*args, **kwargs):
        return None

    class GenerativeModel:
        def __init__(self, *args, **kwargs):
            pass

    genai_stub.configure = configure
    genai_stub.GenerativeModel = GenerativeModel
    sys.modules["google.generativeai"] = genai_stub
    sys.modules["google"].generativeai = genai_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")

    def load_dotenv(*args, **kwargs):
        return None

    dotenv_stub.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv_stub

from services.orchestrator.retrieval_orchestrator import RetrievalOrchestrator
from services.vector.retrieval import VectorRetrieval


class TestHybridRetrieval(unittest.TestCase):
    def test_vector_embedding_uses_deterministic_token_index(self):
        idx_a = VectorRetrieval._stable_index("coffee")
        idx_b = VectorRetrieval._stable_index("coffee")
        self.assertEqual(idx_a, idx_b)
        self.assertGreaterEqual(idx_a, 0)
        self.assertLess(idx_a, VectorRetrieval.EMBEDDING_DIM)

    def test_vector_embedding_is_stable(self):
        vec1 = VectorRetrieval._embed_text("Bought groceries and paid rent")
        vec2 = VectorRetrieval._embed_text("Bought groceries and paid rent")
        self.assertEqual(vec1, vec2)

    def test_vector_embedding_is_normalized(self):
        vec1 = VectorRetrieval._embed_text("Bought groceries and paid rent")
        self.assertEqual(len(vec1), VectorRetrieval.EMBEDDING_DIM)
        norm = sum(x * x for x in vec1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_vector_cosine_similarity_bounds(self):
        a = VectorRetrieval._embed_text("coffee expense")
        b = VectorRetrieval._embed_text("coffee expense")
        c = VectorRetrieval._embed_text("mortgage payment")
        sim_ab = VectorRetrieval._cosine_similarity(a, b)
        sim_ac = VectorRetrieval._cosine_similarity(a, c)
        self.assertGreaterEqual(sim_ab, sim_ac)
        self.assertGreaterEqual(sim_ab, 0.0)
        self.assertLessEqual(sim_ab, 1.0)

    def test_memory_citations_include_graph_and_vector(self):
        orchestrator = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
        graph_context = [
            {
                "type": "Fact",
                "retrieval_score": 0.8,
                "snippet": "Saved $500 this month",
                "properties": {"id": "fact_1", "text": "Saved $500 this month", "confidence": 0.9},
                "score_breakdown": {"hop_distance": 1},
            }
        ]
        vector_context = [
            {
                "type": "TextMemory",
                "retrieval_score": 0.7,
                "snippet": "I usually save on weekends",
                "text": "I usually save on weekends",
                "properties": {"id": "fact_2", "text": "I usually save on weekends", "confidence": 0.8},
            }
        ]

        citations = orchestrator._format_memory_citations(graph_context, vector_context)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["node_type"], "Fact")
        self.assertEqual(citations[1]["node_type"], "TextMemory")


if __name__ == "__main__":
    unittest.main()
