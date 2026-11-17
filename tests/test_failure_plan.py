from __future__ import annotations

import unittest
from dataclasses import replace

import test_retrieval

from legal_graphrag.experiment import SnapshotGraph, paired_interval
from legal_graphrag.models import ProvisionRelation
from legal_graphrag.ontology import ExpansionConfig


class SharedScoringTests(unittest.TestCase):
    def test_shared_formula_can_recover_graph_candidate_suppressed_by_legacy(self):
        retriever = test_retrieval.ArticleRetrievalTests().make_retriever()
        legacy = retriever.retrieve_articles(
            "điều kiện", config=ExpansionConfig(mode="M3"), top_k=1
        )
        shared = retriever.retrieve_articles(
            "điều kiện", config=ExpansionConfig(mode="M3", scoring="shared"), top_k=1
        )
        self.assertEqual(legacy.hits[0].article_id, "a1")
        self.assertEqual(shared.hits[0].article_id, "a3")
        for value in retriever.last_trace["score_components"].values():
            self.assertAlmostEqual(
                value["final_score"],
                0.7 * max(0, value["cosine"]) + 0.15 * value["lexical_overlap"],
            )

    def test_context_budget_can_exclude_top_article_without_overflow(self):
        retriever = test_retrieval.ArticleRetrievalTests().make_retriever()
        result = retriever.retrieve_articles(
            "điều kiện",
            config=ExpansionConfig(mode="M3", scoring="shared", context_chars=5),
            top_k=3,
        )
        self.assertLessEqual(sum(len(h.text) for h in result.hits), 5)
        self.assertNotIn("a3", [h.article_id for h in result.hits])

    def test_scoring_and_budget_validation(self):
        for kwargs in ({"scoring": "unknown"}, {"context_chars": 0}, {"hybrid_pool": 10}):
            with self.assertRaises(ValueError):
                ExpansionConfig(**kwargs)


class GraphControlTests(unittest.TestCase):
    def setUp(self):
        self.relation = ProvisionRelation(
            "a", "b", "CITES", "ca", "cb", "source", "target", 1, "{}"
        )
        self.edges = [
            self.relation,
            replace(
                self.relation,
                source_id="b",
                target_id="c",
                source_chunk_id="cb",
                target_chunk_id="cc",
            ),
        ]
        self.chunks = {"a": ["ca"], "b": ["cb"], "c": ["cc"]}

    def test_reverse_two_hop_preserves_legal_direction(self):
        graph = SnapshotGraph(self.edges, self.chunks)
        self.assertEqual(graph.reachable(["c"], 1), {"b", "c"})
        self.assertEqual(graph.reachable(["c"], 2), {"a", "b", "c"})
        items = graph.expand_articles(["c"], config=ExpansionConfig(mode="M2"))
        self.assertEqual([i.article_id for i in items], ["b", "a"])
        self.assertEqual(items[0].path[0].source_id, "b")
        self.assertEqual(items[0].path[0].target_id, "c")
        self.assertEqual(len(items[1].path), 2)

    def test_shuffled_control_has_no_fabricated_legal_paths(self):
        graph = SnapshotGraph(self.edges, self.chunks, shuffle_seed=42)
        for item in graph.expand_articles(["a"], config=ExpansionConfig(mode="M2")):
            self.assertEqual(item.path, ())
            self.assertTrue(item.structural)
        self.assertEqual(sum(map(len, graph.adjacency.values())), 4)

    def test_paired_bootstrap_identity_and_constant_effect(self):
        self.assertEqual(paired_interval([0, 1], [0, 1])["upper_95"], 0)
        ci = paired_interval([0, 0], [1, 1])
        self.assertEqual((ci["difference"], ci["lower_95"], ci["upper_95"]), (1, 1, 1))
        with self.assertRaises(ValueError):
            paired_interval([], [])


class CorpusParsingTests(unittest.TestCase):
    def test_numpy_style_dict_array_and_reject_unparsed_text(self):
        import csv
        import importlib.util
        import tempfile
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "repair_coverage", Path(__file__).resolve().parents[1] / "scripts/repair_coverage.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.csv"
            for raw, valid in (
                (
                    "[{'aid': 1, 'content_Article': 'first'}\n "
                    "{'aid': 2, 'content_Article': 'second'}]",
                    True,
                ),
                ("[{'aid': 1, 'content_Article': 'first'} MISSING]", False),
            ):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["law_id", "content"])
                    writer.writeheader()
                    writer.writerow({"law_id": "law", "content": raw})
                if valid:
                    self.assertEqual(
                        module.load_corpus(path), {"1": ("law", "first"), "2": ("law", "second")}
                    )
                else:
                    with self.assertRaises(ValueError):
                        module.load_corpus(path)


class HeadingBoundaryTests(unittest.TestCase):
    def test_article_after_uppercase_section_is_not_merged_into_predecessor(self):
        from test_extraction import make_document

        from legal_graphrag.chunking import _section_segments
        from legal_graphrag.structure import parse_structure

        text = (
            "Điều 128. Nội dung trước. Mục 2 ĐẤT NÔNG NGHIỆP "
            "Điều 129. Hạn mức giao đất. Điều 130. Nội dung sau."
        )
        self.assertEqual(
            [label for label, _ in _section_segments(text)], ["Điều 128", "Điều 129", "Điều 130"]
        )
        articles = [p for p in parse_structure(make_document(text)) if p.level == "Article"]
        self.assertEqual(
            [a.id for a in articles], ["d1:article:128", "d1:article:129", "d1:article:130"]
        )
        self.assertNotIn("Hạn mức", articles[0].text)

    def test_chapter_title_and_ordinary_inline_reference(self):
        from legal_graphrag.chunking import _section_segments

        text = (
            "Chương I QUY ĐỊNH CHUNG Điều 1. Thực hiện theo Mục 2 và "
            "Điều 9 của Luật này. Điều 2. Nội dung khác."
        )
        self.assertEqual(
            [label for label, _ in _section_segments(text)], ["Phần mở đầu", "Điều 1", "Điều 2"]
        )


class MappingNormalizationTests(unittest.TestCase):
    def test_only_typographic_differences_are_normalized(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "repair_coverage", Path(__file__).resolve().parents[1] / "scripts/repair_coverage.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.body_text("Sửa đổi như sau:“Điều 4. Nội dung”"),
            module.body_text('Sửa đổi như sau: "Điều 4. Nội dung"'),
        )
        self.assertNotEqual(
            module.body_text("được áp dụng"), module.body_text("không được áp dụng")
        )
        self.assertNotEqual(module.body_text("Điều 4"), module.body_text("Điều 5"))


if __name__ == "__main__":
    unittest.main()
