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


if __name__ == "__main__":
    unittest.main()
