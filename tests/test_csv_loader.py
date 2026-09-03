from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from legal_graphrag.csv_loader import REQUIRED_COLUMNS, stream_documents


class CsvLoaderTests(unittest.TestCase):
    def test_reads_bom_dates_and_large_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legal.csv"
            row = dict.fromkeys(REQUIRED_COLUMNS, "value")
            row.update(
                {
                    "ID": "abc",
                    "Title": "  Văn   bản mẫu  ",
                    "Toàn văn": "Nội dung " * 20_000,
                    "Số hiệu": "01/2026/QĐ-UBND",
                    "Ngày ban hành": "01/02/2026",
                    "Ngày có hiệu lực": "10/02/2026",
                    "Ngày hết hiệu lực": "--",
                    "Cơ quan ban hành": "--",
                }
            )
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow(row)

            result = list(stream_documents(path))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].title, "Văn bản mẫu")
            self.assertEqual(result[0].issued_date, "2026-02-01")
            self.assertEqual(result[0].effective_date, "2026-02-10")
            self.assertEqual(result[0].expiry_date, "")
            self.assertEqual(result[0].agency, "")
            self.assertGreater(len(result[0].full_text), 100_000)

    def test_missing_column_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("ID,Title\n1,Test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                list(stream_documents(path))


if __name__ == "__main__":
    unittest.main()
