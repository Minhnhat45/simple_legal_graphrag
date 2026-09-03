from __future__ import annotations

import unittest

from legal_graphrag.config import Settings
from legal_graphrag.models import GraphNeighbor, QueryFilters, SearchHit
from legal_graphrag.retrieval import HybridRetriever


class FakeOllama:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeGraph:
    def expand(self, seed_document_ids: list[str], *, limit: int) -> list[GraphNeighbor]:
        self.seed_document_ids = seed_document_ids
        return [
            GraphNeighbor(
                document_id="d3",
                title="Văn bản liên quan",
                number="03/2026/QĐ-UBND",
                graph_score=1.0,
                relations=("cited_by",),
                seed_document_ids=("d1",),
            )
        ]


class FakeVectors:
    def query(
        self,
        vector: list[float],
        *,
        filters: QueryFilters,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if document_ids:
            return [
                SearchHit(
                    "c4",
                    "d3",
                    "Văn bản liên quan",
                    "03/2026/QĐ-UBND",
                    "Điều 3",
                    "graph",
                    0.8,
                )
            ]
        return [
            SearchHit("c1", "d1", "Văn bản 1", "01/2026/QĐ-UBND", "Điều 1", "seed 1", 0.95),
            SearchHit("c2", "d1", "Văn bản 1", "01/2026/QĐ-UBND", "Điều 2", "seed 2", 0.90),
            SearchHit("c3", "d2", "Văn bản 2", "02/2026/TT-X", "Điều 1", "seed 3", 0.85),
        ]


class RetrievalTests(unittest.TestCase):
    def test_graph_neighbors_are_semantically_confirmed_and_fused(self) -> None:
        settings = Settings(
            retrieval_top_k=4,
            max_chunks_per_document=2,
            max_context_chars=1000,
        )
        graph = FakeGraph()
        retriever = HybridRetriever(
            settings,
            graph=graph,  # type: ignore[arg-type]
            vectors=FakeVectors(),  # type: ignore[arg-type]
            ollama=FakeOllama(),  # type: ignore[arg-type]
        )
        bundle = retriever.retrieve("Câu hỏi", top_k=4)
        self.assertEqual(graph.seed_document_ids, ["d1", "d2"])
        self.assertIn("d3", {hit.document_id for hit in bundle.hits})
        graph_hit = next(hit for hit in bundle.hits if hit.document_id == "d3")
        self.assertAlmostEqual(graph_hit.final_score, 0.85)
        self.assertEqual(bundle.graph_neighbors[0].document_id, "d3")


if __name__ == "__main__":
    unittest.main()
