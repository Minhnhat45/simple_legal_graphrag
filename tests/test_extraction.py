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


if __name__ == "__main__":
    unittest.main()
