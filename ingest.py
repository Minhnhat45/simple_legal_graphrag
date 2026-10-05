"""Ingest the CSV baseline into Qdrant: Ollama dense vectors + FastEmbed BM25."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

# Make direct invocation use this checkout even without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_graphrag.chunking import chunk_document
from legal_graphrag.config import Settings
from legal_graphrag.csv_loader import stream_documents
from legal_graphrag.llm import OllamaClient
from legal_graphrag.qdrant_store import POINT_NAMESPACE

BM25_MODEL = "Qdrant/bm25"
BM25_OPTIONS = {"disable_stemmer": True, "avg_len": 256.0}


def ingest(
    csv_path: str | Path,
    settings: Settings,
    *,
    collection: str,
    limit: int | None = None,
    recreate: bool = False,
    client: Any,
    embedder: Any,
    bm25: Any,
) -> dict[str, int]:
    """Build a fresh baseline; explicit recreation keeps corpus comparisons repeatable."""
    settings.validate()
    if not collection.strip() or collection == settings.qdrant_collection:
        raise ValueError("Choose a separate baseline collection from QDRANT_COLLECTION")

    documents = stream_documents(csv_path, limit=limit)
    # Validate the input before any collection deletion.
    first = next(documents, None)
    if first is None:
        raise ValueError("CSV contains no documents")
    exists = client.collection_exists(collection)
    if exists and not recreate:
        raise ValueError(f"Collection {collection!r} exists; use --recreate to rebuild it")

    counts = {"documents": 0, "chunks": 0}
    vector_size = None
    seen_ids: set[str] = set()
    for document in chain([first], documents):
        if document.id in seen_ids:
            raise ValueError(f"Duplicate document ID: {document.id!r}")
        seen_ids.add(document.id)
        chunks = chunk_document(
            document, max_chars=settings.chunk_size_chars, overlap=settings.chunk_overlap_chars
        )
        metadata = asdict(document)
        metadata.pop("full_text")
        metadata["document_id"] = metadata.pop("id")
        metadata["document_number"] = metadata.pop("number")
        for start in range(0, len(chunks), settings.embed_batch_size):
            batch = chunks[start : start + settings.embed_batch_size]
            texts = [chunk.text for chunk in batch]
            dense = embedder.embed(texts)
            sparse = list(bm25.embed(texts))
            if len(dense) != len(batch) or len(sparse) != len(batch):
                raise ValueError("Embedding count does not match chunk count")
            size = vector_size or len(dense[0])
            if size <= 0 or any(len(vector) != size for vector in dense):
                raise ValueError("Embedding vectors must have a consistent positive dimension")
            if vector_size is None:
                # Only replace the old baseline after the first embeddings succeed.
                if exists:
                    client.delete_collection(collection)
                client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(size=size, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
                vector_size = size
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(POINT_NAMESPACE, chunk.id)),
                    vector={
                        "dense": vector,
                        "bm25": models.SparseVector(
                            indices=lexical.indices.tolist(), values=lexical.values.tolist()
                        ),
                    },
                    payload={
                        **metadata,
                        "chunk_id": chunk.id,
                        "chunk_index": chunk.index,
                        "section": chunk.section,
                        "text": chunk.text,
                        "embedding_model": settings.ollama_embed_model,
                        "bm25_model": BM25_MODEL,
                        "bm25_options": BM25_OPTIONS,
                    },
                )
                for chunk, vector, lexical in zip(batch, dense, sparse, strict=True)
            ]
            client.upsert(collection_name=collection, points=points, wait=True)
            counts["chunks"] += len(batch)
        counts["documents"] += 1
        print(f"Ingested {counts['documents']} documents / {counts['chunks']} chunks", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--collection", help="Default: QDRANT_COLLECTION + '_baseline'")
    parser.add_argument("--limit", type=int, help="Maximum number of documents")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the baseline")
    args = parser.parse_args()
    settings = Settings.from_env()
    collection = args.collection or f"{settings.qdrant_collection}_baseline"
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if collection == settings.qdrant_collection or not collection.strip():
        parser.error("--collection must be separate from QDRANT_COLLECTION and nonempty")

    from fastembed import SparseTextEmbedding

    bm25 = SparseTextEmbedding(model_name=BM25_MODEL, **BM25_OPTIONS)
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60
    )
    try:
        counts = ingest(
            args.csv_path,
            settings,
            collection=collection,
            limit=args.limit,
            recreate=args.recreate,
            client=client,
            embedder=OllamaClient(settings),
            bm25=bm25,
        )
    finally:
        client.close()
    print(json.dumps({"collection": collection, **counts}))


if __name__ == "__main__":
    main()
