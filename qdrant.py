"""Read an existing Qdrant collection without running the GraphRAG pipeline.

Examples:
    python qdrant.py --collection vn_legal_chunks_baseline
    python qdrant.py --collection vn_legal_chunks_baseline --query "quyền sử dụng đất"
    python qdrant.py --query "thuế thu nhập" --mode dense --limit 5

Settings come from .env, as in ingest.py. Text searches must use the same
OLLAMA_EMBED_MODEL used during ingestion. No query means browse one page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_graphrag.config import Settings
from legal_graphrag.llm import OllamaClient


def query_collection(
    client: QdrantClient,
    collection: str,
    *,
    query: str | None = None,
    mode: str = "auto",
    limit: int = 10,
    query_filter: models.Filter | None = None,
    offset: int | str | None = None,
    embedder: Any = None,
    bm25: Any = None,
) -> dict[str, Any]:
    """Browse or search; auto uses hybrid for the ingest.py baseline, dense otherwise.

    A browse response includes next_offset for fetching subsequent pages. Filters
    use Qdrant's standard Filter model. Pass an OllamaClient for dense searches.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if mode not in {"auto", "dense", "bm25", "hybrid"}:
        raise ValueError("mode must be auto, dense, bm25, or hybrid")
    if query is not None and not query.strip():
        raise ValueError("query must not be blank")
    if query is not None and offset is not None:
        raise ValueError("offset is only supported when browsing")
    if not client.collection_exists(collection):
        raise ValueError(f"Collection {collection!r} does not exist")
    info = client.get_collection(collection)
    if query is None:
        points, next_offset = client.scroll(
            collection_name=collection, scroll_filter=query_filter, limit=limit,
            offset=offset, with_payload=True, with_vectors=False,
        )
        return {
            "collection": collection,
            "next_offset": next_offset,
            "points": [{"id": p.id, "payload": p.payload or {}} for p in points],
        }

    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors or {}
    if mode == "auto":
        mode = "hybrid" if "bm25" in sparse_vectors else "dense"
    requests = []
    candidate_limit = max(20, limit * 3) if mode == "hybrid" else limit
    if mode in {"dense", "hybrid"}:
        using = None
        if isinstance(vectors, dict):
            if "dense" in vectors:
                using = "dense"
            elif len(vectors) == 1:
                using = next(iter(vectors))
            else:
                raise ValueError("Expected one dense vector or a vector named 'dense'")
            vector_config = vectors[using]
        else:
            vector_config = vectors
        if vector_config is None:
            raise ValueError("Collection has no dense vector")
        if embedder is None:
            raise ValueError("An embedder is required for dense or hybrid search")
        dense = embedder.embed([query])[0]
        if len(dense) != vector_config.size:
            raise ValueError("Embedding dimension mismatch; use the ingestion embedding model")
        requests.append(models.Prefetch(
            query=dense, using=using, filter=query_filter, limit=candidate_limit,
        ))
    if mode in {"bm25", "hybrid"}:
        if "bm25" not in sparse_vectors:
            raise ValueError("Collection has no 'bm25' sparse vector")
        if bm25 is None:
            from fastembed import SparseTextEmbedding

            # Match the baseline ingestion's tokenization and model options.
            from ingest import BM25_MODEL, BM25_OPTIONS

            bm25 = SparseTextEmbedding(model_name=BM25_MODEL, **BM25_OPTIONS)
        sparse = next(iter(bm25.query_embed(query)))
        requests.append(models.Prefetch(
            query=models.SparseVector(
                indices=sparse.indices.tolist(), values=sparse.values.tolist(),
            ),
            using="bm25", filter=query_filter, limit=candidate_limit,
        ))
    if mode == "hybrid":
        response = client.query_points(
            collection_name=collection, prefetch=requests,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True, with_vectors=False,
        )
    else:
        response = client.query_points(
            collection_name=collection, query=requests[0].query, using=requests[0].using,
            query_filter=query_filter, limit=limit, with_payload=True, with_vectors=False,
        )
    return {
        "collection": collection,
        "mode": mode,
        "points": [
            {"id": p.id, "score": p.score, "payload": p.payload or {}}
            for p in response.points
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", help="Default: QDRANT_COLLECTION from .env")
    parser.add_argument("--query", help="Search text; omit to browse stored points")
    parser.add_argument("--mode", choices=["auto", "dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", help="Browse cursor from a previous next_offset")
    parser.add_argument("--filter", help='Qdrant filter JSON, e.g. {"must": [...]}')
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.query is not None and (not args.query.strip() or args.offset is not None):
        parser.error("--query must be nonblank and cannot be combined with --offset")
    try:
        query_filter = models.Filter.model_validate_json(args.filter) if args.filter else None
        settings = Settings.from_env()
        collection = args.collection or settings.qdrant_collection
        offset = int(args.offset) if args.offset and args.offset.isdecimal() else args.offset
        client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60,
        )
        try:
            result = query_collection(
                client, collection, query=args.query, mode=args.mode, limit=args.limit,
                query_filter=query_filter, offset=offset, embedder=OllamaClient(settings),
            )
        finally:
            client.close()
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
