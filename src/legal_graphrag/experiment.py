"""Snapshot graph controls and paired diagnostics; gold never enters retrieval."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from .models import ArticleCandidate


class SnapshotGraph:
    def __init__(self, relations, article_chunks, *, shuffle_seed=None):
        self.article_chunks = article_chunks
        self.adjacency = defaultdict(list)
        self.shuffled = shuffle_seed is not None
        targets = [r.target_id for r in relations]
        if self.shuffled:
            random.Random(shuffle_seed).shuffle(targets)
        for relation, target in zip(relations, targets, strict=True):
            source = relation.source_id
            # Endpoint permutation preserves the directed degree sequence. Self-loops
            # and duplicate edges are retained and reported, not selectively redrawn.
            self.adjacency[source].append((target, relation))
            self.adjacency[target].append((source, relation))

    def reachable(self, seeds, depth):
        visited, frontier = set(seeds), set(seeds)
        for _ in range(depth):
            frontier = {
                target for source in frontier for target, _ in self.adjacency[source]
            } - visited
            visited.update(frontier)
        return visited

    def expand_articles(self, seeds, *, config, as_of=""):
        if as_of:
            raise ValueError("Snapshot diagnostic graph does not support temporal filtering")
        if config.mode == "M0":
            return []
        visited = set(seeds[: config.seed_limit])
        frontier = {aid: (1.0, ()) for aid in visited}
        result = []
        for depth in range(config.depth):
            proposals = []
            for source, (parent_score, path) in sorted(frontier.items()):
                edges = [
                    (target, r)
                    for target, r in self.adjacency[source]
                    if r.kind in config.kinds and r.confidence >= config.confidence
                ]
                edges.sort(
                    key=lambda x: (-config.weights.get(x[1].kind, 0) * x[1].confidence, x[0])
                )
                for target, relation in edges[: config.per_node_limit]:
                    if target in visited:
                        continue
                    score = (
                        parent_score * config.weights.get(relation.kind, 0) * relation.confidence
                    )
                    if depth:
                        score *= config.decay
                    # Randomized links are candidates only, never fabricated legal evidence.
                    new_path = () if self.shuffled else (*path, relation)
                    proposals.append(
                        ArticleCandidate(
                            target,
                            tuple(self.article_chunks.get(target, ())),
                            score,
                            new_path,
                            self.shuffled,
                        )
                    )
            frontier = {}
            for item in sorted(proposals, key=lambda x: (-x.score, x.article_id)):
                if item.article_id in visited or item.score <= 0:
                    continue
                visited.add(item.article_id)
                result.append(item)
                frontier[item.article_id] = (item.score, item.path)
                if len(result) >= config.candidate_budget:
                    return result
        return result


def paired_interval(baseline, treatment, *, seed=20261113, repetitions=2000):
    if not baseline or len(baseline) != len(treatment):
        raise ValueError("Require equally sized, nonempty paired samples")
    differences = [b - a for a, b in zip(baseline, treatment, strict=True)]
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)
    )
    return {
        "difference": statistics.mean(differences),
        "lower_95": draws[int(0.025 * repetitions)],
        "upper_95": draws[min(repetitions - 1, int(0.975 * repetitions))],
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def summarize_comparisons(predictions):
    modes = list(dict.fromkeys(r["mode"] for r in predictions))
    baseline = {r["qid"]: r for r in predictions if r["mode"] == "hybrid"}
    result = {}
    for mode in modes:
        rows = [r for r in predictions if r["mode"] == mode]
        result[mode] = {}
        for subset, selected in (
            ("all", rows),
            ("fully_covered", [r for r in rows if r["fully_covered"]]),
        ):
            if not selected:
                result[mode][subset] = {"queries": 0}
                continue
            result[mode][subset] = {
                "queries": len(selected),
                **{
                    metric: statistics.mean(r[metric] for r in selected)
                    for metric in ("recall", "candidate_recall", "complete_evidence", "latency_ms")
                },
                "recovered_gold_occurrences": sum(len(r["recovered"]) for r in selected),
                "lost_gold_occurrences": sum(len(r["lost"]) for r in selected),
                "new_candidate_gold_occurrences": sum(
                    len(r["new_candidate_gold"]) for r in selected
                ),
                "paired_recall": paired_interval(
                    [baseline[r["qid"]]["recall"] for r in selected],
                    [r["recall"] for r in selected],
                ),
                "paired_complete_evidence": paired_interval(
                    [baseline[r["qid"]]["complete_evidence"] for r in selected],
                    [r["complete_evidence"] for r in selected],
                ),
            }
    return result
