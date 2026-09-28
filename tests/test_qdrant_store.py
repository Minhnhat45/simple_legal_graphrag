from __future__ import annotations

import unittest
import warnings

from qdrant_client import QdrantClient

from legal_graphrag.config import Settings
from legal_graphrag.models import DocumentChunk, LegalDocument, QueryFilters
from legal_graphrag.qdrant_store import QdrantStore


class QdrantStoreTests(unittest.TestCase):
    def test_upsert_filtered_query_and_delete_in_memory(self) -> None:
        store = QdrantStore(
            Settings(qdrant_collection="test_chunks"),
            client=QdrantClient(":memory:"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            store.ensure_collection(3)

        document = LegalDocument(
            id="d1",
            title="Văn bản mẫu",
            full_text="Điều 1.",
            number="01/2026/QĐ-X",
            document_type="Quyết định",
            sector="Khoa học",
            issued_date="2026-01-01",
            field="Đo lường",
            effective_date="2026-01-02",
            status="Còn hiệu lực",
            expiry_date="",
            agency="Bộ X",
            signer_position="Bộ trưởng",
            signer="Nguyễn Văn A",
        )
        chunk = DocumentChunk(
            id="d1:0",
            document_id="d1",
            index=0,
            section="Điều 1",
            text="Nội dung",
        )
        store.upsert([chunk], [[1.0, 0.0, 0.0]], {"d1": document})

        hits = store.query(
            [1.0, 0.0, 0.0],
            filters=QueryFilters(status="Còn hiệu lực"),
            limit=3,
        )
        self.assertEqual([hit.chunk_id for hit in hits], ["d1:0"])
        self.assertEqual(store.count(), 1)

        self.assertEqual(store.read_document_chunks(["d1"]), [(chunk, [1.0, 0.0, 0.0])])
        evidence = store.evidence_hits([chunk.id], [1.0, 0.0, 0.0], filters=QueryFilters())
        self.assertEqual([hit.chunk_id for hit in evidence], [chunk.id])
        self.assertEqual(
            store.evidence_hits(
                [chunk.id],
                [1.0, 0.0, 0.0],
                filters=QueryFilters(agency="Other agency"),
            ),
            [],
        )

        store.delete_document_chunks(["d1"])
        self.assertEqual(store.count(), 0)


if __name__ == "__main__":
    unittest.main()
