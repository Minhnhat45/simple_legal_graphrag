"""Article-level extension of the existing VLSP evaluation conventions.

Binary metrics follow evaluate_retrieval.score_ranking. Keep relevant_aids as
original gold IDs; never replace gold articles with their individual chunks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from .models import QueryFilters
from .ontology import ExpansionConfig


def score_ranking(retrieved_ids, relevant_ids, top_k):
    relevant = set(map(str, relevant_ids))
    if not relevant or top_k <= 0:
        raise ValueError("Expected nonempty gold articles and positive top_k")
    ranked = list(dict.fromkeys(map(str, retrieved_ids)))[:top_k]
    hits = [rank for rank, aid in enumerate(ranked, 1) if aid in relevant]
    precision, recall = len(hits) / top_k, len(hits) / len(relevant)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(top_k, len(relevant)) + 1))
    return {
        "hit_rate": float(bool(hits)),
        "precision": precision,
        "recall": recall,
        "mrr": 1 / hits[0] if hits else 0.0,
        "ndcg": sum(1 / math.log2(r + 1) for r in hits) / ideal,
        "f1": 2 * precision * recall / (precision + recall) if hits else 0.0,
    }


def load_eval(path):
    output, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            gold = json.loads(row.get("relevant_aids", row.get("relevant_article_ids", "[]")))
            if (
                not isinstance(gold, list)
                or not gold
                or any(
                    isinstance(a, bool) or not isinstance(a, (int, str)) or not str(a).strip()
                    for a in gold
                )
            ):
                raise ValueError("Gold must be a nonempty list of original Article IDs")
            if not row["question"].strip() or row["qid"] in seen:
                raise ValueError("Blank question or repeated qid")
            seen.add(row["qid"])
            output.append({**row, "gold": list(dict.fromkeys(map(str, gold)))})
    if not output:
        raise ValueError("Empty evaluation")
    return output


def evaluate(retriever, samples, *, top_k=10, config=None, original_ids=None):
    config = config or ExpansionConfig()
    original_ids = original_ids or {}
    predictions, summaries, embeddings = [], defaultdict(list), []
    for sample in samples:
        start = time.monotonic()
        vector = retriever.ollama.embed([sample["question"]])[0]
        embeddings.append((time.monotonic() - start) * 1000)
        # Freeze the actual seed result, not only the query vector. Approximate
        # search and RRF tie ordering can otherwise confound graph ablations.
        start = time.monotonic()
        hybrid_hits = retriever.vectors.hybrid_query(
            sample["question"],
            vector,
            filters=QueryFilters(),
            limit=config.candidate_budget,
            candidate_limit=config.candidate_budget,
        )
        hybrid_ms = (time.monotonic() - start) * 1000
        baseline, baseline_candidates = set(), set()
        for mode in ("M0", "M1", "M2", "M3", "M4"):
            result = retriever.retrieve_articles(
                sample["question"],
                top_k=top_k,
                config=replace(config, mode=mode),
                question_vector=vector,
                hybrid_hits=hybrid_hits,
                hybrid_latency_ms=hybrid_ms,
            )
            ranked = list(
                dict.fromkeys(
                    h.original_article_id or original_ids.get(h.article_id, h.article_id)
                    for h in result.hits
                )
            )[:top_k]
            metrics = score_ranking(ranked, sample["gold"], top_k)
            trace = retriever.last_trace
            candidates = {original_ids.get(a, a) for a in trace["candidate_articles"]}
            gold = set(sample["gold"])
            if mode == "M0":
                baseline = set(ranked)
                baseline_candidates = candidates
            recovered = sorted((set(ranked) & gold) - baseline)
            lost = sorted((baseline & gold) - set(ranked))
            record = {
                "qid": sample["qid"],
                "mode": mode,
                "gold": sample["gold"],
                "retrieved": ranked,
                **metrics,
                "latency_ms": trace["latency_ms"],
                "candidate_recall": len(candidates & gold) / len(gold),
                "recovered_articles": recovered,
                "lost_articles": lost,
                "new_evidence_beyond_baseline_candidates": sorted(
                    (candidates & gold) - baseline_candidates
                ),
                "trace": trace,
            }
            predictions.append(record)
            summaries[mode].append(record)
    overall = {}
    for mode, rows in summaries.items():
        times = sorted(r["latency_ms"] for r in rows)
        overall[mode] = {
            metric: statistics.mean(r[metric] for r in rows)
            for metric in (
                "hit_rate",
                "precision",
                "recall",
                "mrr",
                "ndcg",
                "f1",
                "candidate_recall",
            )
        }
        overall[mode].update(
            {
                "latency_p50_ms": statistics.median(times),
                "latency_p95_ms": times[max(0, math.ceil(len(times) * 0.95) - 1)],
                "recovered_articles": sum(len(r["recovered_articles"]) for r in rows),
                "lost_articles": sum(len(r["lost_articles"]) for r in rows),
            }
        )
    return {
        "samples": len(samples),
        "top_k": top_k,
        "config": asdict(config),
        "overall": overall,
        "embedding_p50_ms": statistics.median(embeddings),
        "timing_note": "One frozen hybrid result per query shared across modes. Each latency "
        "includes measured hybrid time plus the mode's expansion/confirmation/reranking. "
        "Shared embedding timed separately; sparse encoder should be warmed before evaluation.",
        "predictions": predictions,
    }


def save_report(report, output, *, eval_path=None):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if eval_path:
        report["eval_sha256"] = hashlib.sha256(Path(eval_path).read_bytes()).hexdigest()
    # Never overwrite previous experiments, including accidental reruns.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
