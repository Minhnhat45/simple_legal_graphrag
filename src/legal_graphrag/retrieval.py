from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .config import Settings
from .graph_store import GraphStore
from .llm import OllamaClient
from .models import QueryFilters, RetrievalBundle, SearchHit
from .qdrant_store import QdrantStore


class HybridRetriever:
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

    def retrieve(
        self,
        question: str,
        *,
        filters: QueryFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalBundle:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question cannot be empty")
        selected_filters = filters or QueryFilters()
        result_limit = top_k or self.settings.retrieval_top_k
        if result_limit <= 0:
            raise ValueError("top_k must be positive")

        question_vector = self.ollama.embed([cleaned_question])[0]
        seed_hits = self.vectors.query(
            question_vector,
            filters=selected_filters,
            limit=result_limit * 2,
        )
        if not seed_hits:
            return RetrievalBundle(hits=(), graph_neighbors=())

        seed_ids: list[str] = []
        for hit in seed_hits:
            if hit.document_id and hit.document_id not in seed_ids:
                seed_ids.append(hit.document_id)
            if len(seed_ids) >= min(5, result_limit):
                break

        neighbors = self.graph.expand(
            seed_ids,
            limit=self.settings.graph_expansion_limit,
        )
        graph_scores = {neighbor.document_id: neighbor.graph_score for neighbor in neighbors}
        neighbor_hits = (
            self.vectors.query(
                question_vector,
                filters=selected_filters,
                limit=result_limit * 2,
                document_ids=list(graph_scores),
            )
            if graph_scores
            else []
        )

        candidates: dict[str, SearchHit] = {}
        for hit in seed_hits:
            ranked = replace(hit, final_score=max(0.0, hit.vector_score))
            current = candidates.get(hit.chunk_id)
            if current is None or ranked.final_score > current.final_score:
                candidates[hit.chunk_id] = ranked
        for hit in neighbor_hits:
            graph_score = graph_scores.get(hit.document_id, 0.0)
            final_score = 0.75 * max(0.0, hit.vector_score) + 0.25 * graph_score
            ranked = replace(hit, final_score=final_score)
            current = candidates.get(hit.chunk_id)
            if current is None or ranked.final_score > current.final_score:
                candidates[hit.chunk_id] = ranked

        selected: list[SearchHit] = []
        per_document: dict[str, int] = defaultdict(int)
        context_chars = 0
        for hit in sorted(candidates.values(), key=lambda item: item.final_score, reverse=True):
            if per_document[hit.document_id] >= self.settings.max_chunks_per_document:
                continue
            if selected and context_chars + len(hit.text) > self.settings.max_context_chars:
                continue
            selected.append(hit)
            per_document[hit.document_id] += 1
            context_chars += len(hit.text)
            if len(selected) >= result_limit:
                break

        selected_document_ids = {hit.document_id for hit in selected}
        selected_neighbors = tuple(
            neighbor for neighbor in neighbors if neighbor.document_id in selected_document_ids
        )
        return RetrievalBundle(hits=tuple(selected), graph_neighbors=selected_neighbors)
