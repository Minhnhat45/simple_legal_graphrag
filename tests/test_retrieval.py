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


class RelationRetrievalTests(unittest.TestCase):
    def test_exact_evidence_is_prioritized_and_filters_are_forwarded(self) -> None:
        from dataclasses import replace
        from unittest.mock import Mock

        from legal_graphrag.models import LegalRelation

        relation = LegalRelation("d3", "d1", "REPEALS", "c5", "Bãi bỏ", "ollama_verified")
        neighbor = replace(
            FakeGraph().expand([], limit=1)[0], relations=("REPEALS",), evidence=(relation,)
        )
        graph = Mock()
        graph.expand.return_value = [neighbor]
        vectors = Mock(wraps=FakeVectors())
        vectors.evidence_hits = Mock()
        vectors.evidence_hits.return_value = [
            SearchHit("c5", "d3", "title", "number", "Điều 1", "Bãi bỏ", 0.1)
        ]
        retriever = HybridRetriever(
            Settings(max_chunks_per_document=1), graph=graph, vectors=vectors, ollama=FakeOllama()
        )
        filters = QueryFilters(agency="UBND")
        bundle = retriever.retrieve("question", filters=filters, top_k=1)
        self.assertEqual([hit.chunk_id for hit in bundle.hits], ["c5"])
        self.assertEqual(bundle.graph_neighbors[0].evidence, (relation,))
        self.assertEqual(vectors.evidence_hits.call_args.kwargs["filters"], filters)
        vectors.evidence_hits.return_value = []
        bundle = retriever.retrieve("question", top_k=4)
        self.assertTrue(all(not neighbor.evidence for neighbor in bundle.graph_neighbors))
        self.assertTrue(all("REPEALS" not in n.relations for n in bundle.graph_neighbors))

    def test_context_limit_applies_to_first_chunk_too(self) -> None:
        retriever = HybridRetriever(
            Settings(max_context_chars=1),
            graph=FakeGraph(),
            vectors=FakeVectors(),
            ollama=FakeOllama(),
        )
        self.assertEqual(retriever.retrieve("question").hits, ())


if __name__ == "__main__":
    unittest.main()


class ArticleRetrievalTests(unittest.TestCase):
    def make_retriever(self, *, confirmed=True):
        from unittest.mock import Mock

        from legal_graphrag.models import ArticleCandidate, ProvisionRelation

        hits = [
            SearchHit("c1", "d1", "", "", "Điều 1", "nguồn", 0.9, article_id="a1"),
            SearchHit("c2", "d1", "", "", "Điều 1", "nguồn khác", 0.85, article_id="a1"),
            SearchHit("c3", "d2", "", "", "Điều 2", "quy định", 0.8, article_id="a2"),
        ]
        extra = SearchHit("c4", "d3", "", "", "Điều 3", "điều kiện", 0.95, article_id="a3")
        relation = ProvisionRelation(
            "a1", "a3", "SPECIFIES", "c1", "c4", "nguồn", "điều kiện", 0.95, "{}"
        )
        graph = Mock()
        graph.expand_articles.side_effect = lambda ids, config, as_of: (
            [] if config.mode == "M0" else [ArticleCandidate("a3", ("c4",), 1, (relation,))]
        )
        vectors = Mock()
        vectors.hybrid_query.return_value = hits
        all_hits = {h.chunk_id: h for h in hits + ([extra] if confirmed else [])}
        vectors.evidence_hits.side_effect = lambda ids, v, filters: [
            all_hits[cid] for cid in ids if cid in all_hits
        ]
        return HybridRetriever(Settings(), graph=graph, vectors=vectors, ollama=FakeOllama())

    def test_deduplicates_chunks_and_retains_original_hybrid_order(self):
        from legal_graphrag.ontology import ExpansionConfig

        retriever = self.make_retriever()
        result = retriever.retrieve_articles("câu hỏi", config=ExpansionConfig(mode="M0"), top_k=3)
        self.assertEqual([h.article_id for h in result.hits], ["a1", "a2"])
        self.assertEqual(retriever.last_trace["graph_articles"], {})

    def test_graph_can_add_article_only_after_both_evidence_quotes_confirmed(self):
        from legal_graphrag.ontology import ExpansionConfig

        for confirmed in (True, False):
            retriever = self.make_retriever(confirmed=confirmed)
            filters = QueryFilters(agency="Cơ quan")
            result = retriever.retrieve_articles(
                "điều kiện", config=ExpansionConfig(mode="M3"), top_k=3, filters=filters
            )
            self.assertEqual("a3" in {h.article_id for h in result.hits}, confirmed)
            for call in retriever.vectors.evidence_hits.call_args_list:
                self.assertEqual(call.kwargs["filters"], filters)
            self.assertLessEqual(len(retriever.last_trace["candidate_articles"]), 60)

    def test_metrics_keep_gold_articles_and_count_each_article_once(self):
        from legal_graphrag.evaluation import score_ranking

        metrics = score_ranking(["1", "1", "2"], [1, 2], 2)
        self.assertTrue(all(v == 1 for v in metrics.values()))
        metrics = score_ranking(["1"], [1, 2], 2)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        with self.assertRaises(ValueError):
            score_ranking(["1"], [], 2)

    def test_evaluation_freezes_hybrid_seeds_across_all_modes(self):
        from legal_graphrag.evaluation import evaluate

        retriever = self.make_retriever()
        result = evaluate(
            retriever, [{"qid": "q", "question": "điều kiện", "gold": ["a3"]}], top_k=3
        )
        retriever.vectors.hybrid_query.assert_called_once()
        seeds = [r["trace"]["hybrid_articles"] for r in result["predictions"]]
        self.assertTrue(all(s == seeds[0] for s in seeds))
        self.assertEqual(result["overall"]["M0"]["recall"], 0)
        self.assertEqual(result["overall"]["M3"]["recall"], 1)

    def test_configuration_rejects_unbounded_expansion_and_invalid_weights(self):
        from legal_graphrag.ontology import ExpansionConfig

        for values in (
            {"depth": 5},
            {"confidence": float("nan")},
            {"candidate_budget": 0},
            {"weights": {"CITES": -1}},
            {"mode": "unverified"},
        ):
            with self.assertRaises(ValueError):
                ExpansionConfig(**values)
