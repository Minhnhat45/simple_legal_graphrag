from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from .config import Settings
from .models import DocumentChunk, LegalDocument, QueryFilters, SearchHit

BM25_MODEL = "Qdrant/bm25"
BM25_OPTIONS = {"disable_stemmer": True, "avg_len": 256.0}

POINT_NAMESPACE = uuid.UUID("681661fb-803a-42b3-947f-c4bedfe6909f")


class QdrantStore:
    INDEXED_PAYLOAD_FIELDS = (
        "document_id",
        "document_number",
        "document_type",
        "sector",
        "field",
        "status",
        "agency",
    )

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self.models = models
        self.collection = settings.qdrant_collection
        self.hybrid = settings.qdrant_hybrid
        self._ensured_vector_size: int | None = None
        self._vector_layout = None
        self._bm25 = None
        self.article_mapping: dict[str, str] = {}
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def document_chunk_counts(self, document_ids: Sequence[str]) -> dict[str, int]:
        """Check this collection before trusting a shared Neo4j ingestion marker."""
        return {
            did: self.client.count(
                self.collection,
                count_filter=self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key="document_id", match=self.models.MatchValue(value=did)
                        )
                    ]
                ),
                exact=True,
            ).count
            for did in document_ids
        }

    def reset(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection)
        self._ensured_vector_size = None
        self._vector_layout = None

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("Embedding vector size must be positive")
        if self._ensured_vector_size == vector_size:
            return
        if not self.collection_exists():
            config = self.models.VectorParams(
                size=vector_size, distance=self.models.Distance.COSINE
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": config} if self.hybrid else config,
                **(
                    {
                        "sparse_vectors_config": {
                            "bm25": self.models.SparseVectorParams(
                                modifier=self.models.Modifier.IDF
                            )
                        }
                    }
                    if self.hybrid
                    else {}
                ),
            )
        else:
            info = self.client.get_collection(self.collection)
            if self.hybrid and "bm25" not in (info.config.params.sparse_vectors or {}):
                raise ValueError(
                    "Existing collection is dense-only; use a NEW collection for hybrid"
                )
            vectors = info.config.params.vectors
            existing_size = getattr(vectors, "size", None)
            if existing_size is None and isinstance(vectors, dict):
                existing_size = getattr(next(iter(vectors.values())), "size", None)
            if existing_size is not None and int(existing_size) != vector_size:
                raise ValueError(
                    "Qdrant collection vector size does not match the configured embedding "
                    f"model ({existing_size} != {vector_size}). Re-ingest with --recreate."
                )

        for field_name in self.INDEXED_PAYLOAD_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=self.models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                message = str(exc).casefold()
                if "already" not in message and "exists" not in message:
                    raise
        self._ensured_vector_size = vector_size

    def delete_document_chunks(self, document_ids: Sequence[str]) -> None:
        if not document_ids or not self.collection_exists():
            return
        selector = self.models.FilterSelector(
            filter=self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key="document_id",
                        match=self.models.MatchAny(any=list(document_ids)),
                    )
                ]
            )
        )
        self.client.delete(
            collection_name=self.collection,
            points_selector=selector,
            wait=True,
        )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        documents: dict[str, LegalDocument],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        from .structure import map_chunks

        mappings = {
            cid: aid
            for doc in documents.values()
            for cid, aid in map_chunks(doc, [c for c in chunks if c.document_id == doc.id]).items()
        }
        named = self._dense_name()
        has_sparse = "bm25" in (self._layout().sparse_vectors or {})
        sparse = list(self._sparse_encoder().embed([c.text for c in chunks])) if has_sparse else []
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            document = documents[chunk.document_id]
            point_id = str(uuid.uuid5(POINT_NAMESPACE, chunk.id))
            points.append(
                self.models.PointStruct(
                    id=point_id,
                    vector=(
                        {
                            named: list(vector),
                            **(
                                {
                                    "bm25": self.models.SparseVector(
                                        indices=sparse[len(points)].indices.tolist(),
                                        values=sparse[len(points)].values.tolist(),
                                    )
                                }
                                if has_sparse
                                else {}
                            ),
                        }
                        if named
                        else list(vector)
                    ),
                    payload={
                        "chunk_id": chunk.id,
                        "article_id": mappings.get(chunk.id, ""),
                        "chunk_index": chunk.index,
                        "section": chunk.section,
                        "text": chunk.text,
                        "document_id": document.id,
                        "title": document.title,
                        "document_number": document.number,
                        "document_type": document.document_type,
                        "sector": document.sector,
                        "field": document.field,
                        "status": document.status,
                        "agency": document.agency,
                        "issued_date": document.issued_date,
                        "effective_date": document.effective_date,
                        "expiry_date": document.expiry_date,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def _filter(
        self,
        filters: QueryFilters,
        document_ids: Sequence[str] | None,
    ) -> Any | None:
        conditions = [
            self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
            for key, value in filters.as_payload_map().items()
        ]
        if document_ids:
            conditions.append(
                self.models.FieldCondition(
                    key="document_id",
                    match=self.models.MatchAny(any=list(document_ids)),
                )
            )
        return self.models.Filter(must=conditions) if conditions else None

    def query(
        self,
        vector: Sequence[float],
        *,
        filters: QueryFilters,
        limit: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        if not self.collection_exists():
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=self._dense_name(),
            query_filter=self._filter(filters, document_ids),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    title=str(payload.get("title", "")),
                    number=str(payload.get("document_number", "")),
                    section=str(payload.get("section", "")),
                    text=str(payload.get("text", "")),
                    vector_score=float(point.score),
                    article_id=self.article_mapping.get(
                        str(payload.get("chunk_id", point.id)), payload.get("article_id", "")
                    ),
                    original_article_id=str(payload.get("original_article_id", "")),
                )
            )
        return hits

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def health(self) -> bool:
        self.client.get_collections()
        return True

    def read_document_chunks(
        self,
        document_ids: Sequence[str],
    ) -> list[tuple[DocumentChunk, list[float]]]:
        """Read stored vectors without invoking the embedding model."""
        if not document_ids or not self.collection_exists():
            return []
        output = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(QueryFilters(), document_ids),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                vector = point.vector
                if isinstance(vector, dict):
                    vector = vector.get(self._dense_name())
                if not isinstance(vector, list) or not vector:
                    raise ValueError(f"Missing dense vector for point {point.id}")
                output.append(
                    (
                        DocumentChunk(
                            id=payload["chunk_id"],
                            document_id=payload["document_id"],
                            index=payload["chunk_index"],
                            section=payload["section"],
                            text=payload["text"],
                        ),
                        vector,
                    )
                )
            if offset is None:
                return sorted(output, key=lambda item: (item[0].document_id, item[0].index))

    def evidence_hits(
        self,
        chunk_ids: Sequence[str],
        vector: Sequence[float],
        *,
        filters: QueryFilters,
    ) -> list[SearchHit]:
        """Confirm exact evidence in Qdrant using the query vector and user filters."""
        if not chunk_ids or not self.collection_exists():
            return []
        query_filter = self._filter(filters, None) or self.models.Filter(must=[])
        query_filter.must = list(query_filter.must or []) + [
            self.models.HasIdCondition(
                has_id=[str(uuid.uuid5(POINT_NAMESPACE, chunk_id)) for chunk_id in chunk_ids]
            )
        ]
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=self._dense_name(),
            query_filter=query_filter,
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchHit(
                chunk_id=p.payload["chunk_id"],
                document_id=p.payload["document_id"],
                title=p.payload["title"],
                number=p.payload["document_number"],
                section=p.payload["section"],
                text=p.payload["text"],
                vector_score=float(p.score),
                article_id=self.article_mapping.get(
                    p.payload["chunk_id"], p.payload.get("article_id", "")
                ),
                original_article_id=str(p.payload.get("original_article_id", "")),
            )
            for p in response.points
        ]

    def _layout(self):
        if self._vector_layout is None:
            self._vector_layout = self.client.get_collection(self.collection).config.params
        return self._vector_layout

    def _dense_name(self):
        vectors = self._layout().vectors
        if not isinstance(vectors, dict):
            return None
        if "dense" in vectors:
            return "dense"
        if len(vectors) == 1:
            return next(iter(vectors))
        raise ValueError("Expected one dense vector or a vector named dense")

    def hybrid_query(self, question, vector, *, filters, limit, candidate_limit=None):
        """Reuse ingest.py's named dense/BM25 layout and Qdrant RRF fusion."""
        if not self.collection_exists():
            return []
        if "bm25" not in (self._layout().sparse_vectors or {}):
            raise ValueError(
                "Hybrid mode requires a BM25 collection; choose the existing baseline "
                "or build a separate collection with ingest.py"
            )
        sparse = next(iter(self._sparse_encoder().query_embed(question)))
        query_filter = self._filter(filters, None)
        budget = candidate_limit or max(20, limit * 3)
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                self.models.Prefetch(
                    query=list(vector), using=self._dense_name(), filter=query_filter, limit=budget
                ),
                self.models.Prefetch(
                    query=self.models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                    using="bm25",
                    filter=query_filter,
                    limit=budget,
                ),
            ],
            query=self.models.FusionQuery(fusion=self.models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        from .structure import article_id

        return [
            SearchHit(
                chunk_id=str(p.payload.get("chunk_id", p.id)),
                document_id=p.payload["document_id"],
                title=p.payload.get("title", ""),
                number=p.payload.get("document_number", ""),
                section=p.payload.get("section", ""),
                text=p.payload["text"],
                vector_score=float(p.score),
                article_id=self.article_mapping.get(
                    str(p.payload.get("chunk_id", p.id)),
                    p.payload.get(
                        "article_id",
                        article_id(p.payload["document_id"], p.payload.get("section", "")),
                    ),
                ),
                original_article_id=str(p.payload.get("original_article_id", "")),
            )
            for p in response.points
        ]

    def _sparse_encoder(self):
        if self._bm25 is None:
            from fastembed import SparseTextEmbedding

            self._bm25 = SparseTextEmbedding(model_name=BM25_MODEL, **BM25_OPTIONS)
        return self._bm25
