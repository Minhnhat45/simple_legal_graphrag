from __future__ import annotations

import unittest

from legal_graphrag.config import Settings
from legal_graphrag.models import RetrievalBundle, SearchHit
from legal_graphrag.service import GraphRAGService


class FakeRetriever:
    def retrieve(self, question: str, *, filters: object, top_k: int | None) -> RetrievalBundle:
        return RetrievalBundle(
            hits=(
                SearchHit(
                    chunk_id="d1:0",
                    document_id="d1",
                    title="Văn bản mẫu",
                    number="01/2026/QĐ-UBND",
                    section="Điều 1",
                    text="Điều 1 quy định phạm vi áp dụng.",
                    vector_score=0.9,
                    final_score=0.9,
                ),
            ),
            graph_neighbors=(),
        )


class FakeOllama:
    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return "Phạm vi áp dụng được quy định tại Điều 1 [S1]."


class ServiceTests(unittest.TestCase):
    def test_answer_marks_sources_actually_cited(self) -> None:
        ollama = FakeOllama()
        service = GraphRAGService(
            Settings(),
            graph=object(),  # type: ignore[arg-type]
            vectors=object(),  # type: ignore[arg-type]
            ollama=ollama,  # type: ignore[arg-type]
            retriever=FakeRetriever(),  # type: ignore[arg-type]
        )
        result = service.answer("Phạm vi áp dụng là gì?")
        self.assertTrue(result.sources[0].cited)
        self.assertEqual(result.warnings, ())
        self.assertIn("NGỮ CẢNH ĐƯỢC TRUY XUẤT", ollama.messages[1]["content"])

    def test_missing_citation_produces_warning(self) -> None:
        class NoCitationOllama(FakeOllama):
            def chat(self, messages: list[dict[str, str]]) -> str:
                return "Phạm vi áp dụng được quy định tại Điều 1."

        service = GraphRAGService(
            Settings(),
            graph=object(),  # type: ignore[arg-type]
            vectors=object(),  # type: ignore[arg-type]
            ollama=NoCitationOllama(),  # type: ignore[arg-type]
            retriever=FakeRetriever(),  # type: ignore[arg-type]
        )
        result = service.answer("Phạm vi áp dụng là gì?")
        self.assertEqual(len(result.warnings), 1)

    def test_graph_claim_uses_supporting_chunk_citation_and_preserves_direction(self) -> None:
        from dataclasses import replace
        from unittest.mock import Mock

        from legal_graphrag.models import GraphNeighbor, LegalRelation

        bundle = FakeRetriever().retrieve("question", filters=None, top_k=None)
        relation = LegalRelation(
            "d1",
            "d2",
            "REPEALS",
            "d1:0",
            "Điều 1 quy định phạm vi áp dụng.",
            "test",
        )
        neighbor = GraphNeighbor("d2", "Target", "02", 1.0, ("REPEALS",), ("d1",), (relation,))
        retriever = Mock()
        retriever.retrieve.return_value = replace(bundle, graph_neighbors=(neighbor, neighbor))
        ollama = FakeOllama()
        service = GraphRAGService(
            Settings(), graph=object(), vectors=object(), ollama=ollama, retriever=retriever
        )
        service.answer("question")
        prompt = ollama.messages[1]["content"]
        self.assertEqual(prompt.count("[S1] d1 --REPEALS--> d2"), 1)
        retriever.retrieve.return_value = replace(
            bundle,
            graph_neighbors=(
                replace(
                    neighbor,
                    evidence=(replace(relation, source_chunk_id="missing"),),
                ),
            ),
        )
        service.answer("question")
        self.assertNotIn("REPEALS", ollama.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
