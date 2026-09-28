from __future__ import annotations

import unittest
from dataclasses import replace
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


class RelationBuilderTests(unittest.TestCase):
    def test_missing_and_outdated_documents_fail_before_model_calls(self) -> None:
        from unittest.mock import Mock

        from legal_graphrag.relation_builder import RelationBuilder

        graph, vectors, ollama = Mock(), Mock(), Mock()
        graph.relation_catalog.return_value = {
            "sample-001": {"status": "building", "index_hash": "old"},
        }
        builder = RelationBuilder(Settings(), graph=graph, vectors=vectors, ollama=ollama)
        path = Path(__file__).resolve().parents[1] / "data/sample_legal_documents.csv"
        with self.assertRaisesRegex(ValueError, "missing=.*sample-002.*outdated=.*sample-001"):
            builder.build(path)
        ollama.chat.assert_not_called()
        vectors.read_document_chunks.assert_not_called()

    def test_sparse_lexical_small_corpus_fallback_and_body_only(self) -> None:
        from legal_graphrag.csv_loader import stream_documents
        from legal_graphrag.models import DocumentChunk
        from legal_graphrag.relation_builder import Candidate, RelationBuilder

        path = Path(__file__).resolve().parents[1] / "data/sample_legal_documents.csv"
        documents = list(stream_documents(path))[:2]
        chunks = [
            DocumentChunk(
                d.id + ":0",
                d.id,
                0,
                "Điều 1",
                "Identical header\nSố hiệu: number\nMục: Điều 1\n" + body,
            )
            for d, body in zip(documents, ("hồ sơ", "hồ sơ"), strict=True)
        ]
        pairs = {}

        def get_pair(source, target):
            return pairs.setdefault((source, target), Candidate(source, target))

        builder = RelationBuilder(Settings(), graph=object(), vectors=object(), ollama=object())
        builder._lexical(chunks, documents, get_pair)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p.ranks["lexical"] == 1 for p in pairs.values()))
        pairs.clear()
        chunks = [
            replace(c, text=c.text.replace("hồ sơ", body))
            for c, body in zip(chunks, ("alpha", "beta"), strict=True)
        ]
        builder._lexical(chunks, documents, get_pair)
        self.assertEqual(pairs, {})

    def test_build_keeps_explicit_pairs_beyond_cap_and_rejects_ambiguous_targets(self) -> None:
        import json
        from unittest.mock import Mock

        from legal_graphrag.chunking import chunk_document
        from legal_graphrag.csv_loader import stream_documents
        from legal_graphrag.extraction import normalize_document_number
        from legal_graphrag.relation_builder import RelationBuilder

        settings = Settings(relation_pair_limit=1)
        path = Path(__file__).resolve().parents[1] / "data/sample_legal_documents.csv"
        docs = list(stream_documents(path))
        chunks = [
            c
            for d in docs
            for c in chunk_document(
                d,
                max_chars=settings.chunk_size_chars,
                overlap=settings.chunk_overlap_chars,
            )
        ]
        graph, vectors, ollama = Mock(), Mock(), Mock()
        ingestor = Ingestor(settings, graph=graph, vectors=vectors, ollama=ollama)
        graph.relation_catalog.return_value = {
            d.id: {
                "status": "ready",
                "index_hash": ingestor._index_hash(d),
                "chunk_count": sum(c.document_id == d.id for c in chunks),
            }
            for d in docs
        }
        graph.identifier_owners.return_value = {
            normalize_document_number(d.number): [d.id] for d in docs
        }
        graph.entity_candidates.side_effect = lambda source, limit: [
            (d.id, 1) for d in docs if d.id != source
        ]
        vectors.read_document_chunks.return_value = [(c, [1.0, 0.0]) for c in chunks]
        vectors.query.return_value = []
        ollama.chat.return_value = json.dumps({"kind": "NONE"})
        builder = RelationBuilder(settings, graph=graph, vectors=vectors, ollama=ollama)
        report = builder.build(path)
        self.assertEqual(report["shortlisted"], 4)
        self.assertEqual(report["accepted_relations"], 1)
        self.assertEqual(report["ollama_calls"], 4)
        citation = graph.upsert_relations.call_args.args[0][0]
        self.assertEqual(
            (citation.source_id, citation.target_id, citation.kind),
            ("sample-003", "sample-001", "CITES"),
        )
        graph.identifier_owners.return_value["01/2026/QĐ-UBND"].append("duplicate")
        report = builder.build(path)
        self.assertEqual(report["accepted_relations"], 0)
        self.assertIn(
            {"source_id": "sample-003", "number": "01/2026/QĐ-UBND"},
            report["unresolved_references"],
        )
        vectors.read_document_chunks.return_value = [(c, [1.0, 0.0]) for c in chunks[:-1]]
        with self.assertRaisesRegex(ValueError, "Missing/outdated Qdrant chunks.*sample-003"):
            builder.build(path)


if __name__ == "__main__":
    unittest.main()
