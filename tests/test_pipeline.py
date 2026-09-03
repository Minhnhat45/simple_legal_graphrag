from __future__ import annotations

import unittest
from pathlib import Path

from legal_graphrag.config import Settings
from legal_graphrag.pipeline import Ingestor


class FakeGraph:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}
        self.chunk_ids: list[str] = []

    def reset(self) -> None:
        self.hashes.clear()
        self.chunk_ids.clear()

    def ensure_schema(self) -> None:
        pass

    def get_index_hashes(self, document_ids: list[str]) -> dict[str, str]:
        return {key: self.hashes[key] for key in document_ids if key in self.hashes}

    def prepare_documents(self, documents: list[object], mentions: list[object]) -> None:
        self.prepared = getattr(self, "prepared", 0) + len(documents)

    def upsert_chunks(self, chunks: list[object]) -> None:
        self.chunk_ids.extend(chunk.id for chunk in chunks)

    def mark_indexed(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            self.hashes[str(row["id"])] = str(row["index_hash"])


class FakeVectors:
    def __init__(self) -> None:
        self.exists = False
        self.points = 0

    def collection_exists(self) -> bool:
        return self.exists

    def reset(self) -> None:
        self.exists = False
        self.points = 0

    def ensure_collection(self, vector_size: int) -> None:
        self.exists = True
        self.vector_size = vector_size

    def delete_document_chunks(self, document_ids: list[str]) -> None:
        pass

    def upsert(
        self,
        chunks: list[object],
        vectors: list[list[float]],
        documents: dict[str, object],
    ) -> None:
        self.points += len(chunks)


class FakeOllama:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.5] for text in texts]


class PipelineTests(unittest.TestCase):
    def test_second_identical_ingest_is_skipped(self) -> None:
        root = Path(__file__).resolve().parents[1]
        csv_path = root / "data" / "sample_legal_documents.csv"
        settings = Settings(
            chunk_size_chars=500,
            chunk_overlap_chars=50,
            embed_batch_size=2,
            document_batch_size=2,
        )
        graph = FakeGraph()
        vectors = FakeVectors()
        ingestor = Ingestor(
            settings,
            graph=graph,  # type: ignore[arg-type]
            vectors=vectors,  # type: ignore[arg-type]
            ollama=FakeOllama(),  # type: ignore[arg-type]
        )

        first = ingestor.ingest(csv_path, recreate=True)
        second = ingestor.ingest(csv_path)
        self.assertEqual(first.documents_indexed, 3)
        self.assertGreater(first.chunks_indexed, 0)
        self.assertEqual(second.documents_indexed, 0)
        self.assertEqual(second.documents_skipped, 3)
        self.assertEqual(len(graph.hashes), 3)


if __name__ == "__main__":
    unittest.main()
