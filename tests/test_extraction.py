from __future__ import annotations

import unittest

from legal_graphrag.extraction import extract_mentions, normalize_document_number
from legal_graphrag.models import LegalDocument


def make_document(text: str) -> LegalDocument:
    return LegalDocument(
        id="d1",
        title="Văn bản mẫu",
        full_text=text,
        number="01/2026/QĐ-UBND",
        document_type="Quyết định",
        sector="Khoa học và công nghệ",
        issued_date="2026-01-01",
        field="Quản lý khoa học",
        effective_date="2026-01-02",
        status="Còn hiệu lực",
        expiry_date="",
        agency="UBND tỉnh Mẫu",
        signer_position="Chủ tịch",
        signer="Nguyễn Văn A",
    )


class ExtractionTests(unittest.TestCase):
    def test_normalizes_number_spacing_and_case(self) -> None:
        self.assertEqual(
            normalize_document_number(" 46 / 2008 / qđ - ubnd "),
            "46/2008/QĐ-UBND",
        )

    def test_extracts_and_aggregates_legal_mentions(self) -> None:
        mentions = extract_mentions(
            make_document(
                "Căn cứ Luật số 29/2013/QH13. Bãi bỏ Quyết định số 46/2008/QĐ-UBND. "
                "Quyết định 46/2008/QĐ-UBND được nhắc lại. "
                "Số hiệu của văn bản này là 01/2026/QĐ-UBND."
            )
        )
        by_number = {item.number_norm: item for item in mentions}
        self.assertIn("29/2013/QH13", by_number)
        self.assertEqual(by_number["46/2008/QĐ-UBND"].count, 2)
        self.assertIn("repeals", by_number["46/2008/QĐ-UBND"].kinds)
        self.assertNotIn("01/2026/QĐ-UBND", by_number)

    def test_dates_are_not_mistaken_for_document_numbers(self) -> None:
        mentions = extract_mentions(make_document("Ban hành ngày 01/02/2026."))
        self.assertEqual(mentions, [])


class RelationEvidenceTests(unittest.TestCase):
    def test_reference_offsets_exclude_repeated_headers(self) -> None:
        from legal_graphrag.extraction import chunk_body, extract_reference_spans
        from legal_graphrag.models import DocumentChunk

        text = (
            "Title mentions 29/2013/QH13\nsecond title line\n"
            "Số hiệu: 03/2026/QĐ-UBND\nMục: Điều 1\n"
            "Bãi bỏ Quyết định số 01 / 2026 / QĐ-UBND."
        )
        chunk = DocumentChunk("source:0", "source", 0, "Điều 1", text)
        refs = extract_reference_spans(chunk, "03/2026/QĐ-UBND")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].number_norm, "01/2026/QĐ-UBND")
        self.assertEqual(text[refs[0].start : refs[0].end], refs[0].text)
        self.assertTrue(chunk_body(text).startswith("Bãi bỏ"))

    def test_verifier_rejects_malformed_wrong_target_and_header_evidence(self) -> None:
        import json

        from legal_graphrag.models import DocumentChunk
        from legal_graphrag.relation_builder import Candidate, validate_relation

        body = "Bãi bỏ Quyết định số 01/2026/QĐ-UBND."
        chunk = DocumentChunk(
            "s:0",
            "s",
            0,
            "Điều 1",
            "Misleading header\nSố hiệu: 03/2026/QĐ-UBND\nMục: Điều 1\n" + body,
        )
        pair = Candidate("s", "t")
        valid = dict(
            source_id="s", target_id="t", kind="REPEALS", source_chunk_id="s:0", evidence=body
        )

        def verify(data):
            return validate_relation(json.dumps(data), pair, [chunk], [], "01/2026/QĐ-UBND")

        self.assertEqual(verify(valid).kind, "REPEALS")
        for changes in (
            {"target_id": "other"},
            {"kind": "DEFINES"},
            {"source_chunk_id": "t:0"},
            {"target_chunk_id": "fake"},
            {"evidence": "Misleading header"},
            {"evidence": "Invented quote"},
            {"evidence": "01/2026/QĐ-UBND"},
            {"kind": "AMENDS"},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(verify({**valid, **changes}))
        for data in ([], None, {"kind": "NONE"}):
            self.assertIsNone(verify(data))
        self.assertIsNone(validate_relation("not JSON", pair, [chunk], [], "01/2026/QĐ-UBND"))


if __name__ == "__main__":
    unittest.main()
