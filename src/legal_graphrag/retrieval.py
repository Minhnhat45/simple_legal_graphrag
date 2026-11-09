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

        evidence_ids = list(
            dict.fromkeys(
                relation.source_chunk_id for neighbor in neighbors for relation in neighbor.evidence
            )
        )
        evidence_hits = (
            self.vectors.evidence_hits(
                evidence_ids,
                question_vector,
                filters=selected_filters,
            )
            if evidence_ids
            else []
        )
        confirmed_evidence = {hit.chunk_id for hit in evidence_hits}
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

        for hit in evidence_hits:
            candidates[hit.chunk_id] = replace(hit, final_score=max(0.0, hit.vector_score))

        selected: list[SearchHit] = []
        per_document: dict[str, int] = defaultdict(int)
        context_chars = 0
        for hit in sorted(
            candidates.values(),
            key=lambda item: (item.chunk_id in confirmed_evidence, item.final_score),
            reverse=True,
        ):
            if per_document[hit.document_id] >= self.settings.max_chunks_per_document:
                continue
            if context_chars + len(hit.text) > self.settings.max_context_chars:
                continue
            selected.append(hit)
            per_document[hit.document_id] += 1
            context_chars += len(hit.text)
            if len(selected) >= result_limit:
                break

        selected_document_ids = {hit.document_id for hit in selected}
        selected_chunk_ids = {hit.chunk_id for hit in selected}
        selected_neighbors = tuple(
            replace(
                neighbor,
                evidence=tuple(
                    relation
                    for relation in neighbor.evidence
                    if relation.source_chunk_id in selected_chunk_ids
                ),
                relations=tuple(
                    label
                    for label in neighbor.relations
                    if label.islower()
                    or any(
                        relation.kind == label and relation.source_chunk_id in selected_chunk_ids
                        for relation in neighbor.evidence
                    )
                ),
            )
            for neighbor in neighbors
            if neighbor.document_id in selected_document_ids
        )
        return RetrievalBundle(hits=tuple(selected), graph_neighbors=selected_neighbors)

    def retrieve_articles(
        self,
        question,
        *,
        config=None,
        filters=None,
        top_k=None,
        as_of="",
        question_vector=None,
        hybrid_hits=None,
        hybrid_latency_ms=0.0,
    ):
        """Hybrid -> Article seeds -> verified paths -> confirmation -> bounded reranking.

        Every mode uses the same hybrid candidate limit and final candidate budget.
        One representative chunk is returned per article; all path evidence must be
        present in Qdrant under the same filters before the path can affect ranking.
        """
        import math
        import re
        import time

        from .models import GraphNeighbor, LegalRelation
        from .ontology import ExpansionConfig

        config = config or ExpansionConfig()
        filters = filters or QueryFilters()
        k = self.settings.retrieval_top_k if top_k is None else top_k
        if not question.strip() or k <= 0 or k > config.candidate_budget:
            raise ValueError("Require nonblank question and 0 < top_k <= candidate_budget")
        if as_of:
            from datetime import date

            date.fromisoformat(as_of)
        started = time.monotonic()
        vector = (
            question_vector if question_vector is not None else self.ollama.embed([question])[0]
        )
        seeds = (
            hybrid_hits
            if hybrid_hits is not None
            else self.vectors.hybrid_query(
                question,
                vector,
                filters=filters,
                limit=config.candidate_budget,
                candidate_limit=config.candidate_budget,
            )
        )

        # Never infer an article for a preamble or an ambiguous repeated heading.
        def identity(hit):
            return hit.article_id

        by_article = {}
        for hit in seeds:
            aid = identity(hit)
            if aid and aid not in by_article:
                by_article[aid] = replace(hit, article_id=aid)
        seed_articles = list(by_article)[: config.seed_limit]
        expanded = self.graph.expand_articles(seed_articles, config=config, as_of=as_of)
        # Reserve slots for graph discoveries, then enforce one shared candidate cap.
        expanded = expanded[: max(0, config.candidate_budget - len(seed_articles))]
        proposal_ids = list(dict.fromkeys(cid for item in expanded for cid in item.chunk_ids))
        # Bound confirmation work independently of article length and graph degree.
        proposal_ids = proposal_ids[: config.candidate_budget * 2]
        path_ids = list(
            dict.fromkeys(
                cid
                for item in expanded
                for r in item.path
                for cid in (r.source_chunk_id, r.target_chunk_id)
            )
        )
        confirmation_ids = list(dict.fromkeys(proposal_ids + path_ids))
        confirmed = self.vectors.evidence_hits(confirmation_ids, vector, filters=filters)
        hit_map = {hit.chunk_id: hit for hit in confirmed}
        graph_scores, paths, kinds = {}, {}, {}
        for item in expanded:
            if any(
                r.source_chunk_id not in hit_map
                or r.target_chunk_id not in hit_map
                or r.evidence not in hit_map[r.source_chunk_id].text
                or r.target_evidence not in hit_map[r.target_chunk_id].text
                for r in item.path
            ):
                continue
            hits = [hit_map[cid] for cid in item.chunk_ids if cid in hit_map]
            if not hits:
                continue
            best = max(hits, key=lambda h: (h.vector_score, h.chunk_id))
            by_article[item.article_id] = replace(best, article_id=item.article_id)
            graph_scores[item.article_id] = item.score
            paths[item.article_id] = item.path
            kinds[item.article_id] = [r.kind for r in item.path] or ["STRUCTURAL"]
        selected_ids = list(
            dict.fromkeys(
                seed_articles
                + sorted(graph_scores, key=lambda aid: (-graph_scores[aid], aid))
                + list(by_article)
            )
        )[: config.candidate_budget]
        candidates = [by_article[aid] for aid in selected_ids]
        # Query-specific cosine confirmation for ALL candidates avoids mixing RRF and cosine.
        confirmed_all = self.vectors.evidence_hits(
            [h.chunk_id for h in candidates], vector, filters=filters
        )
        cosine = {h.chunk_id: h.vector_score for h in confirmed_all}
        seed_ranks = {identity(h): rank for rank, h in reversed(list(enumerate(seeds, 1)))}
        tokens = set(re.findall(r"\w+", question.casefold()))
        ranked = []
        for hit in candidates:
            if hit.chunk_id not in cosine or not math.isfinite(cosine[hit.chunk_id]):
                continue
            # M0 keeps the existing hybrid order. Other modes only adjust articles
            # connected by a verified path; raw graph evidence never gets forced first.
            hybrid = 1 / (60 + seed_ranks[hit.article_id]) if hit.article_id in seed_ranks else 0
            score = 61 * hybrid
            if hit.article_id in graph_scores:
                overlap = len(tokens & set(re.findall(r"\w+", hit.text.casefold()))) / max(
                    1, len(tokens)
                )
                score = max(score, 0.7 * max(0, cosine[hit.chunk_id]) + 0.15 * overlap)
                score += config.graph_weight * graph_scores.get(hit.article_id, 0)
            ranked.append(replace(hit, vector_score=cosine[hit.chunk_id], final_score=score))
        ranked.sort(key=lambda h: (-h.final_score, h.article_id))
        hits = tuple(ranked[:k])
        selected_chunks = {h.chunk_id for h in hits}
        neighbors = []
        for hit in hits:
            path = paths.get(hit.article_id, ())
            grounded = [r for r in path if r.source_chunk_id in selected_chunks]
            if grounded:
                neighbors.append(
                    GraphNeighbor(
                        hit.document_id,
                        hit.title,
                        hit.number,
                        graph_scores[hit.article_id],
                        tuple(r.kind for r in grounded),
                        (),
                        tuple(
                            LegalRelation(
                                hit_map[r.source_chunk_id].document_id,
                                hit_map[r.target_chunk_id].document_id,
                                r.kind,
                                r.source_chunk_id,
                                r.evidence,
                                r.method,
                                r.target_chunk_id,
                                r.provenance,
                            )
                            for r in grounded
                        ),
                    )
                )
        self.last_trace = {
            "seed_articles": seed_articles,
            "hybrid_articles": list(seed_ranks),
            "candidate_articles": selected_ids,
            "graph_articles": graph_scores,
            "relation_kinds": kinds,
            "paths": {aid: [r.kind for r in path] for aid, path in paths.items()},
            "latency_ms": (time.monotonic() - started) * 1000 + hybrid_latency_ms,
        }
        return RetrievalBundle(hits=hits, graph_neighbors=tuple(neighbors))
