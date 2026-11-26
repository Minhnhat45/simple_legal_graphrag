"""Validate complete predictions and report matched controls and failure stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_graphrag.experiment import paired_interval


def summarize(snapshot, comparison, output):
    coverage = json.loads((snapshot / "coverage.json").read_text())
    evaluation = json.loads((comparison / "evaluation.json").read_text())
    rows = [
        json.loads(line) for line in (comparison / "predictions.jsonl").read_text().splitlines()
    ]
    modes = list(evaluation["configurations"])
    expected = {(r["qid"], mode) for r in coverage["diagnostics"] for mode in modes}
    keyed = {(r["qid"], r["mode"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("Incomplete, unexpected or duplicated prediction records")
    if rows != evaluation["predictions"]:
        raise ValueError("JSONL predictions differ from final evaluation")
    for row in rows:
        if len(row["retrieved"]) > 10 or len(row["retrieved"]) != len(set(row["retrieved"])):
            raise ValueError("Invalid top-K articles")
        if sum(len(h["text"]) for h in row["selected_evidence"]) != row["trace"]["context_chars"]:
            raise ValueError("Saved evidence does not match timed retrieval context")
        if row["trace"]["context_chars"] > 18000:
            raise ValueError("Exceeded context budget")
        if (
            len(row["trace"]["confirmed_candidate_articles"])
            > evaluation["configurations"][row["mode"]]["candidate_budget"]
        ):
            raise ValueError("Exceeded candidate cap")
    comparisons = {}
    for baseline, treatment in [
        ("hybrid", "hybrid_larger"),
        ("hybrid", "hybrid_shared"),
        ("hybrid_shared", "graph_shared"),
        ("shuffled_shared", "graph_shared"),
        ("graph_shared", "graph_shared_path"),
        ("graph_legacy", "graph_shared"),
    ]:
        comparisons[f"{treatment}_minus_{baseline}"] = {}
        for subset in ("all", "fully_covered"):
            selected = [
                r
                for r in rows
                if r["mode"] == treatment and (subset == "all" or r["fully_covered"])
            ]
            comparisons[f"{treatment}_minus_{baseline}"][subset] = {
                "queries": len(selected),
                **{
                    metric: paired_interval(
                        [keyed[r["qid"], baseline][metric] for r in selected],
                        [r[metric] for r in selected],
                    )
                    for metric in ("recall", "candidate_recall", "complete_evidence")
                },
            }
    distributions = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        distributions[mode] = {}
        for metric, values in {
            "retrieval_ms": [r["latency_ms"] for r in selected],
            "embedding_ms": [r["embedding_ms"] for r in selected],
            "confirmed_articles": [
                len(r["trace"]["confirmed_candidate_articles"]) for r in selected
            ],
            "hybrid_pool_unique_articles": [len(r["trace"]["hybrid_articles"]) for r in selected],
            "context_chars": [r["trace"]["context_chars"] for r in selected],
        }.items():
            values.sort()
            distributions[mode][metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95_nearest_rank": values[max(0, (95 * len(values) + 99) // 100 - 1)],
                "min": min(values),
                "max": max(values),
            }
    originals = coverage["original_ids"]
    stages = []
    for sample in coverage["diagnostics"]:
        gold = set(sample["gold"])
        base = keyed[sample["qid"], "hybrid"]
        graph = keyed[sample["qid"], "graph_shared"]
        hybrid_candidates = {
            originals.get(a, a) for a in base["trace"]["confirmed_candidate_articles"]
        }
        graph_candidates = {
            originals.get(a, a) for a in graph["trace"]["confirmed_candidate_articles"]
        }
        stages.append(
            {
                "qid": sample["qid"],
                "question": sample["question"],
                "gold": sorted(gold),
                "mapped": sample["snapshot_mapped_gold"],
                "unmapped": sample["snapshot_unmapped_gold"],
                "hybrid_candidate_gold": sorted(gold & hybrid_candidates),
                "one_hop_gold": graph["reachability"]["1"]["retrievable_gold"],
                "two_hop_gold": graph["reachability"]["2"]["retrievable_gold"],
                "reachable_beyond_hybrid": sorted(
                    set(graph["reachability"]["2"]["retrievable_gold"]) - hybrid_candidates
                ),
                "graph_candidate_gold": sorted(gold & graph_candidates),
                "new_graph_candidate_gold": sorted((gold & graph_candidates) - hybrid_candidates),
                "gold_ranks": {
                    mode: {
                        a: r["retrieved"].index(a) + 1 if a in r["retrieved"] else None
                        for a in gold
                    }
                    for mode in modes
                    for r in [keyed[sample["qid"], mode]]
                },
                "answer_used_correctly": None,
            }
        )
    report = {
        "status": "complete_diagnostic_retrieval; independent_review_pending",
        "records": len(rows),
        "queries": len(coverage["diagnostics"]),
        "evaluation_sha256": hashlib.sha256(
            (comparison / "evaluation.json").read_bytes()
        ).hexdigest(),
        "snapshot_sha256": coverage["snapshot_sha256"],
        "coverage": {
            k: coverage[k]
            for k in [
                "documents",
                "articles",
                "chunks",
                "mapped_original_articles",
                "resolved_citations",
                "snapshot_fully_mapped_queries",
                "snapshot_mapped_gold_occurrences",
                "gold_occurrences",
                "graph_build_seconds",
                "build_seconds",
            ]
        },
        "mapping_recall_ceiling": statistics.mean(
            len(row["snapshot_mapped_gold"]) / len(row["gold"])
            for row in coverage["diagnostics"]
        ),
        "overall": evaluation["overall"],
        "paired_controls": comparisons,
        "actual_budgets_and_latency": distributions,
        "reachability": {
            "gold_occurrences_reachable_beyond_hybrid_candidates": sum(
                len(r["reachable_beyond_hybrid"]) for r in stages
            ),
            "gold_occurrences_added_by_graph": sum(
                len(r["new_graph_candidate_gold"]) for r in stages
            ),
        },
        "stages": stages,
        "index_seconds": evaluation["index_seconds"],
        "embedding_digest": evaluation["embedding_digest"],
        "collection": evaluation["collection"],
        "limitations": [
            "Existing 100 questions are diagnostic; no untouched-test performance claim.",
            "Coverage and candidate caps are matched upper bounds; realized counts differ.",
            "Gold article identity does not prove selected chunks contain the required clauses.",
            "Shared scorer is cosine plus lexical overlap, not a trained cross-encoder.",
            "Citations are automatically resolved; independent precision review is pending.",
            "Retrieval latency includes shared hybrid lookup; query embedding is separate.",
            "Bootstrap intervals are paired, exploratory and not multiplicity adjusted.",
        ],
    }
    with output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({m: evaluation["overall"][m]["all"] for m in modes}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "comparison", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    summarize(a.snapshot, a.comparison, a.output)
