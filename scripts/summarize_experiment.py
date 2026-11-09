"""Summarize frozen M0–M4 results without changing queries, rankings or gold IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_graphrag.evaluation import save_report


def summarize(evaluation, manifest, relations):
    rows = evaluation["predictions"]
    originals = manifest["original_ids"]
    available = set(originals.values())
    documents = {original: aid.rsplit(":article:", 1)[0] for aid, original in originals.items()}
    subsets = {
        "fully_covered": lambda row: set(row["gold"]) <= available,
        "multiple_gold_articles": lambda row: len(row["gold"]) > 1,
        "covered_cross_document_gold": lambda row: (
            set(row["gold"]) <= available and len({documents[a] for a in row["gold"]}) > 1
        ),
    }
    result = {
        "samples": evaluation["samples"],
        "top_k": evaluation["top_k"],
        "config": evaluation["config"],
        "coverage": evaluation["coverage"],
        "eval_sha256": evaluation["eval_sha256"],
        "overall": evaluation["overall"],
        "embedding_p50_ms": evaluation["embedding_p50_ms"],
        "relations": {
            k: v for k, v in relations.items() if k not in {"relations", "verification_audit"}
        },
        "cross_document_edges": dict(
            Counter(
                r["kind"]
                for r in relations["relations"]
                if r["source_id"].rsplit(":article:", 1)[0]
                != r["target_id"].rsplit(":article:", 1)[0]
            )
        ),
        "subsets": {},
        "graph_contribution": {},
    }
    audit = relations["verification_audit"]
    result["verification_outcomes"] = {
        "supported": sum(r.get("review", {}).get("supported") is True for r in audit),
        "explicit_NONE": sum(r.get("review", {}).get("supported") is False for r in audit),
        "malformed_or_empty": sum("review" in r and not r["review"] for r in audit),
        "clause_scope_rejections": sum("reason" in r for r in audit),
    }
    for name, include in subsets.items():
        output = {}
        for mode in evaluation["overall"]:
            selected = [r for r in rows if r["mode"] == mode and include(r)]
            output[mode] = {"queries": len(selected)}
            if selected:
                output[mode].update(
                    {
                        metric: statistics.mean(r[metric] for r in selected)
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
                )
        result["subsets"][name] = output
    for mode in evaluation["overall"]:
        selected = [r for r in rows if r["mode"] == mode]
        kind_counts = Counter()
        recovered = []
        for row in selected:
            for article in row["recovered_articles"]:
                kinds = sorted(
                    {
                        kind
                        for aid, path in row["trace"]["relation_kinds"].items()
                        if originals.get(aid, aid) == article
                        for kind in path
                    }
                )
                kind_counts.update(kinds)
                recovered.append({"qid": row["qid"], "article": article, "path_kinds": kinds})
        result["graph_contribution"][mode] = {
            "queries_with_expansion": sum(bool(r["trace"]["graph_articles"]) for r in selected),
            "new_gold_candidate_occurrences": sum(
                len(r["new_evidence_beyond_baseline_candidates"]) for r in selected
            ),
            "new_gold_candidates": [
                {"qid": r["qid"], "articles": r["new_evidence_beyond_baseline_candidates"]}
                for r in selected
                if r["new_evidence_beyond_baseline_candidates"]
            ],
            "recovered_by_path_kind": dict(kind_counts),
            "recovered": recovered,
            "lost": [
                {"qid": r["qid"], "articles": r["lost_articles"]}
                for r in selected
                if r["lost_articles"]
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [args.evaluation, args.manifest, args.relations]
    result = summarize(*(json.loads(Path(p).read_text()) for p in paths))
    result["inputs"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    save_report(result, args.output)


if __name__ == "__main__":
    main()
