from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from .chunking import chunk_document
from .config import Settings
from .csv_loader import stream_documents
from .extraction import extract_mentions
from .graph_store import GraphStore
from .llm import OllamaClient
from .models import DocumentChunk, IngestReport, LegalDocument
from .qdrant_store import QdrantStore

INDEX_SCHEMA_VERSION = "vn-legal-graphrag/v1"
T = TypeVar("T")


def batched(values: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        *,
        graph: GraphStore | None = None,
        vectors: QdrantStore | None = None,
        ollama: OllamaClient | None = None,
    ) -> None:
        self.settings = settings
        self.graph = graph or GraphStore(settings)
        self.vectors = vectors or QdrantStore(settings)
        self.ollama = ollama or OllamaClient(settings)

    def _index_hash(self, document: LegalDocument) -> str:
        signature = "|".join(
            (
                INDEX_SCHEMA_VERSION,
                document.fingerprint,
                self.settings.ollama_embed_model,
                str(self.settings.chunk_size_chars),
                str(self.settings.chunk_overlap_chars),
            )
        )
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def _embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        progress: Callable[[str], None] | None = None,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for chunk_batch in batched(chunks, self.settings.embed_batch_size):
            vectors.extend(self.ollama.embed([chunk.text for chunk in chunk_batch]))
            completed = len(vectors)
            interval = self.settings.embed_batch_size * 10
            if progress and (completed == len(chunks) or completed % interval == 0):
                progress(f"embedding_chunks={completed}/{len(chunks)}")
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        return vectors

    def ingest(
        self,
        csv_path: str | Path,
        *,
        limit: int | None = None,
        recreate: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> IngestReport:
        source = str(Path(csv_path).expanduser().resolve())
        if recreate:
            self.graph.reset()
            self.vectors.reset()
        self.graph.ensure_schema()

        seen = indexed = skipped = chunk_count = mention_count = 0
        documents = stream_documents(source, limit=limit)
        for document_batch in batched(documents, self.settings.document_batch_size):
            seen += len(document_batch)
            desired_hashes = {
                document.id: self._index_hash(document) for document in document_batch
            }
            current_hashes = self.graph.get_index_hashes(list(desired_hashes))
            qdrant_available = self.vectors.collection_exists()
            pending = [
                document
                for document in document_batch
                if not qdrant_available
                or current_hashes.get(document.id) != desired_hashes[document.id]
            ]
            skipped += len(document_batch) - len(pending)
            if not pending:
                if progress:
                    progress(f"documents_seen={seen} indexed={indexed} skipped={skipped}")
                continue

            chunks = [
                chunk
                for document in pending
                for chunk in chunk_document(
                    document,
                    max_chars=self.settings.chunk_size_chars,
                    overlap=self.settings.chunk_overlap_chars,
                )
            ]
            embedded = self._embed_chunks(chunks, progress)
            if not embedded or not embedded[0]:
                raise RuntimeError("The embedding model returned no vectors")
            self.vectors.ensure_collection(len(embedded[0]))

            mentions = [mention for document in pending for mention in extract_mentions(document)]
            self.graph.prepare_documents(pending, mentions)
            self.vectors.delete_document_chunks([document.id for document in pending])

            document_map = {document.id: document for document in pending}
            write_batch_size = max(self.settings.embed_batch_size * 4, 32)
            for start in range(0, len(chunks), write_batch_size):
                end = start + write_batch_size
                chunk_slice = chunks[start:end]
                self.vectors.upsert(chunk_slice, embedded[start:end], document_map)
                self.graph.upsert_chunks(chunk_slice)

            counts_by_document = {document.id: 0 for document in pending}
            for chunk in chunks:
                counts_by_document[chunk.document_id] += 1
            self.graph.mark_indexed(
                [
                    {
                        "id": document.id,
                        "index_hash": desired_hashes[document.id],
                        "chunk_count": counts_by_document[document.id],
                    }
                    for document in pending
                ]
            )

            indexed += len(pending)
            chunk_count += len(chunks)
            mention_count += len(mentions)
            if progress:
                progress(
                    f"documents_seen={seen} indexed={indexed} skipped={skipped} "
                    f"chunks_indexed={chunk_count}"
                )

        return IngestReport(
            source=source,
            documents_seen=seen,
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_indexed=chunk_count,
            mentions_indexed=mention_count,
        )
