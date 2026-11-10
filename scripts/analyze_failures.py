"""Audit frozen retrieval results at original-article and exact law_id levels.

Uses only saved artifacts and the original evaluation/corpus CSVs; no database calls.
Law hits are measured within the saved top-K articles, not a new top-K-law ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ORG_PATTERN = (
    r"vị trí và chức năng|cơ cấu tổ chức|hệ thống tổ chức|trực thuộc|nhiệm vụ và quyền hạn"
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    return {
        "queries": len(rows),
        "article_zero_hits": sum(not r["hit_articles"] for r in rows),
        "article_incomplete": sum(bool(r["missing_articles"]) for r in rows),
        "law_zero_hits": sum(not r["hit_laws"] for r in rows),
        "law_incomplete": sum(bool(r["missing_laws"]) for r in rows),
        "article_recall": sum(len(r["hit_articles"]) / len(r["gold_articles"]) for r in rows)
        / len(rows)
        if rows
        else None,
        "law_recall": sum(len(r["hit_laws"]) / len(r["gold_laws"]) for r in rows) / len(rows)
        if rows
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", default="reports/runs/20261107/frozen_seeds/evaluation.json"
    )
    parser.add_argument("--manifest", default="reports/runs/20261107/manifest.json")
    parser.add_argument("--queries", default="../data_for_legal_rag/VLSP_legal_data/eval.csv")
    parser.add_argument("--corpus", default="../data_for_legal_rag/VLSP_legal_data/corpus.csv")
    parser.add_argument("--coverage", default="data/coverage.csv")
    parser.add_argument("--output-prefix", default="reports/failure_analysis_20261107")
    args = parser.parse_args()
    csv.field_size_limit(100_000_000)
    evaluation = json.loads(Path(args.evaluation).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    hashes = {
        key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
        for key in ("evaluation", "manifest", "queries", "corpus", "coverage")
    }
    assert hashes["queries"] == evaluation["eval_sha256"], "Evaluation CSV changed"
    queries = {r["qid"]: r for r in read_csv(args.queries)}
    assert len(queries) == evaluation["samples"]
    aid_to_law = {}
    # The corpus uses numpy-style object arrays, not JSON: extract explicit aid fields.
    for row in read_csv(args.corpus):
        for aid in re.findall(r"['\"]aid['\"]:\s*(\d+)", row["content"]):
            assert aid not in aid_to_law, f"Duplicate original aid: {aid}"
            aid_to_law[aid] = row["law_id"]
    originals = manifest["original_ids"]
    indexed_articles = set(originals.values())
    indexed_laws = {aid_to_law[a] for a in indexed_articles}
    coverage = {r["law_id"]: r["status"] for r in read_csv(args.coverage)}
    rows = []
    for prediction in evaluation["predictions"]:
        query = queries[prediction["qid"]]
        gold = set(prediction["gold"])
        assert gold == set(map(str, json.loads(query["relevant_aids"])))
        ranked = prediction["retrieved"]
        candidates = {originals.get(a, a) for a in prediction["trace"]["candidate_articles"]}
        assert gold | set(ranked) | candidates <= aid_to_law.keys(), "Unresolved article identity"
        gold_laws = {aid_to_law[a] for a in gold}
        ranked_laws = list(dict.fromkeys(aid_to_law[a] for a in ranked))
        candidate_laws = {aid_to_law[a] for a in candidates}
        missing_laws = gold_laws - set(ranked_laws)
        law_reasons = {}
        for law in sorted(missing_laws):
            law_reasons[law] = (
                "no_local_source"
                if coverage[law] != "document_found"
                else "law_not_indexed"
                if law not in indexed_laws
                else "law_not_in_candidates"
                if law not in candidate_laws
                else "law_candidate_outside_top_k"
            )
        article_reasons = {
            a: "article_not_indexed"
            if a not in indexed_articles
            else "article_not_in_candidates"
            if a not in candidates
            else "article_candidate_outside_top_k"
            for a in sorted(gold - set(ranked))
        }
        rows.append(
            {
                "qid": prediction["qid"],
                "mode": prediction["mode"],
                "group": query["group"],
                "question": query["question"],
                "organization_query": bool(re.search(ORG_PATTERN, query["question"].lower())),
                "gold_articles": sorted(gold),
                "gold_laws": sorted(gold_laws),
                "retrieved_articles": ranked,
                "retrieved_laws": ranked_laws,
                "hit_articles": sorted(gold & set(ranked)),
                "hit_laws": sorted(gold_laws & set(ranked_laws)),
                "missing_articles": sorted(gold - set(ranked)),
                "missing_laws": sorted(missing_laws),
                "all_gold_articles_indexed": gold <= indexed_articles,
                "all_gold_laws_indexed": gold_laws <= indexed_laws,
                "candidate_gold_articles": sorted(gold & candidates),
                "candidate_gold_laws": sorted(gold_laws & candidate_laws),
                "article_failure_reasons": article_reasons,
                "law_failure_reasons": law_reasons,
                "local_source_status": {law: coverage[law] for law in sorted(gold_laws)},
                "top_three": [{"aid": a, "law_id": aid_to_law[a]} for a in ranked[:3]],
            }
        )
    modes = list(evaluation["overall"])
    assert len(rows) == len(queries) * len(modes)
    assert len({(r["qid"], r["mode"]) for r in rows}) == len(rows)
    summaries = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        assert (
            abs(summarize(selected)["article_recall"] - evaluation["overall"][mode]["recall"])
            < 1e-12
        )
        summaries[mode] = {"overall": summarize(selected), "groups": {}}
        for group in sorted({r["group"] for r in rows}):
            subset = [r for r in selected if r["group"] == group]
            summaries[mode]["groups"][group] = {
                "all": summarize(subset),
                "all_articles_indexed": summarize(
                    [r for r in subset if r["all_gold_articles_indexed"]]
                ),
                "all_laws_indexed": summarize([r for r in subset if r["all_gold_laws_indexed"]]),
            }
    selected = [r for r in rows if r["mode"] == "M4"]
    failures = [r for r in selected if r["missing_laws"]]
    reasons = Counter(reason for r in failures for reason in r["law_failure_reasons"].values())
    article_reasons = Counter(
        reason for r in selected for reason in r["article_failure_reasons"].values()
    )
    law_counts = Counter(law for r in failures for law in r["missing_laws"])
    org = [r for r in selected if r["organization_query"]]
    per_law = [
        {
            "law_id": law,
            "missed_queries": count,
            "target_queries": sum(law in r["gold_laws"] for r in selected),
            "qids": [r["qid"] for r in failures if law in r["missing_laws"]],
        }
        for law, count in law_counts.most_common()
    ]
    same_failures = all(
        {
            (r["qid"], tuple(r["missing_articles"]), tuple(r["missing_laws"]))
            for r in rows
            if r["mode"] == mode
        }
        == {(r["qid"], tuple(r["missing_articles"]), tuple(r["missing_laws"])) for r in selected}
        for mode in modes
    )
    result = {
        "inputs": {
            key: {"path": str(Path(getattr(args, key)).resolve()), "sha256": digest}
            for key, digest in hashes.items()
        },
        "definitions": {
            "law_hit": (
                "Any saved top-K article has exactly the gold corpus law_id; "
                "K counts articles, not unique laws."
            ),
            "law_zero_hits": "No target law appears in the saved top-K articles.",
            "law_incomplete": (
                "At least one target law is missing, including partial multi-law successes."
            ),
            "coverage": (
                "Indexed means represented in the frozen manifest original_ids, "
                "not merely a graph node or local source."
            ),
            "organization_query": (
                f"Exploratory wording-based tag, not an official dataset label: {ORG_PATTERN}"
            ),
            "identity_limit": (
                "Exact corpus law numbers are used; shared numbers do not establish "
                "authority/version identity."
            ),
        },
        "top_k_articles": evaluation["top_k"],
        "mode_summaries": summaries,
        "same_missing_articles_and_laws_all_modes": same_failures,
        "M4_law_failure_occurrences": dict(reasons),
        "M4_article_failure_occurrences": dict(article_reasons),
        "M4_missed_laws": per_law,
        "M4_organization_queries": summarize(org),
        "rows": rows,
    }
    prefix = Path(args.output_prefix)
    targets = [prefix.with_suffix(ext) for ext in (".json", ".csv")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Use a new output prefix; existing analyses are preserved")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with targets[1].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )
    print(
        json.dumps(
            {
                "outputs": list(map(str, targets)),
                "summaries": summaries,
                "law_failure_stages": dict(reasons),
                "article_failure_stages": dict(article_reasons),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
