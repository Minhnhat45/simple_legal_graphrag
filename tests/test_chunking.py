from __future__ import annotations

import unittest

from legal_graphrag.chunking import chunk_document
from legal_graphrag.models import LegalDocument


def document(full_text: str) -> LegalDocument:
    return LegalDocument(
        id="doc-1",
        title="Quyết định thử nghiệm",
        full_text=full_text,
        number="01/2026/QĐ-UBND",
        document_type="Quyết định",
        sector="Khoa học và công nghệ",
        issued_date="2026-01-01",
        field="Quản lý khoa học",
        effective_date="2026-01-10",
        status="Còn hiệu lực",
        expiry_date="",
        agency="UBND tỉnh Mẫu",
        signer_position="Chủ tịch",
        signer="Nguyễn Văn A",
    )


class ChunkingTests(unittest.TestCase):
    def test_article_boundaries_are_preserved(self) -> None:
        chunks = chunk_document(
            document(
                "Phần căn cứ của văn bản. Điều 1. Phạm vi điều chỉnh. "
                "Nội dung thứ nhất. Điều 2: Trách nhiệm thi hành. Nội dung thứ hai."
            ),
            max_chars=500,
            overlap=50,
        )
        self.assertEqual([chunk.section for chunk in chunks], ["Phần mở đầu", "Điều 1", "Điều 2"])
        self.assertIn("Điều 1", chunks[1].text)
        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])

    def test_long_article_is_split_with_stable_ids(self) -> None:
        text = "Điều 1. " + "Nội dung quy định rất dài. " * 80
        chunks = chunk_document(document(text), max_chars=420, overlap=60)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0].id, "doc-1:0")
        self.assertEqual(chunks[-1].index, len(chunks) - 1)
        self.assertTrue(all(chunk.section == "Điều 1" for chunk in chunks))

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(document("Nội dung"), max_chars=100, overlap=100)

    def test_inline_article_reference_is_not_treated_as_a_heading(self) -> None:
        chunks = chunk_document(
            document(
                "Căn cứ quy định tại Điều 10 Nghị định số 12/2026/NĐ-CP. "
                "Điều 1. Phạm vi điều chỉnh."
            ),
            max_chars=500,
            overlap=50,
        )
        self.assertEqual([chunk.section for chunk in chunks], ["Phần mở đầu", "Điều 1"])


if __name__ == "__main__":
    unittest.main()


class HierarchyTests(unittest.TestCase):
    def test_preserves_parent_ids_and_rejects_ambiguous_article_mapping(self):
        from test_extraction import make_document

        from legal_graphrag.models import DocumentChunk
        from legal_graphrag.structure import map_chunks, parse_structure

        doc = make_document(
            "Chương I. Điều 1. Quy định. 1. Phải nộp hồ sơ; a) Bản chính. Điều 2. Trách nhiệm."
        )
        nodes = parse_structure(doc)
        by_level = {p.level: p for p in nodes}
        self.assertEqual(by_level["Point"].parent_id, by_level["Clause"].id)
        self.assertEqual(by_level["Clause"].parent_id, "d1:article:1")
        self.assertEqual(by_level["Article"].parent_id, by_level["Chapter"].id)
        duplicate = make_document("Điều 1. Nội dung. Điều 1. Văn bản kèm theo.")
        chunk = DocumentChunk("d1:0", "d1", 0, "Điều 1", "body")
        self.assertEqual(map_chunks(duplicate, [chunk]), {})
        articles = [p for p in parse_structure(duplicate) if p.level == "Article"]
        self.assertEqual(len({p.id for p in articles}), 2)

    def test_inline_chapter_reference_does_not_reparent_articles(self):
        from test_extraction import make_document

        from legal_graphrag.structure import parse_structure

        nodes = parse_structure(
            make_document(
                "Chương I. Điều 1. Thực hiện theo Chương II của văn bản này. Điều 2. Trách nhiệm."
            )
        )
        self.assertEqual(sum(p.level == "Chapter" for p in nodes), 1)
        self.assertTrue(
            all(p.parent_id.endswith("chapter:I:1") for p in nodes if p.level == "Article")
        )
