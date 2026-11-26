"""Freeze source-based review packets before inspecting retrieval outcomes.

No automatic review decision or held-out performance claim is made. Natural
questions are selected independently of graph success from the supplied train CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from repair_coverage import load_corpus


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare(snapshot, corpus_path, train_path, output):
    output.mkdir(parents=True, exist_ok=False)
    coverage = json.loads((snapshot / "coverage.json").read_text())
    corpus = load_corpus(corpus_path)
    db = sqlite3.connect(f"file:{snapshot / 'snapshot.sqlite'}?mode=ro", uri=True)
    documents = {k: json.loads(v) for k, v in db.execute("SELECT id,payload FROM documents")}
    articles = {
        k: {"document_id": d, "text": t}
        for k, d, t in db.execute("SELECT id,document_id,text FROM articles")
    }
    relations = [json.loads(r[0]) for r in db.execute("SELECT payload FROM relations")]
    originals = coverage["original_ids"]
    inverse = {v: k for k, v in originals.items()}
    diagnostics = coverage["diagnostics"]
    seed = 20261118
    selected = sorted(
        diagnostics, key=lambda r: hashlib.sha256(f"{seed}:{r['qid']}".encode()).hexdigest()
    )[:20]
    save(
        output / "answer_protocol.json",
        {
            "status": "frozen_before_retrieval_results",
            "selection": "20 smallest SHA256(seed:qid)",
            "seed": seed,
            "qids": [r["qid"] for r in selected],
            "snapshot_sha256": coverage["snapshot_sha256"],
            "context_chars": 18000,
            "modes": [
                "hybrid",
                "hybrid_larger",
                "hybrid_shared",
                "graph_shared",
                "graph_shared_path",
                "graph_legacy",
                "shuffled_shared",
            ],
            "generator": "gemma4:e4b",
            "judge": "qwen3.5:9b",
            "generation": {
                "temperature": 0,
                "seed": seed,
                "num_predict": 1024,
                "num_ctx": 16384,
                "think": False,
            },
            "judgment": "Automated, blinded to retrieval mode; human correctness remains pending.",
            "reference": (
                "Full corpus gold provisions, unchanged gold IDs; no reference answer supplied."
            ),
        },
    )
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in relations:
        key = (
            "within_document"
            if articles[r["source_id"]]["document_id"] == articles[r["target_id"]]["document_id"]
            else "cross_document"
        )
        strata[key].append(r)
    relation_items = []
    for stratum, rows in sorted(strata.items()):
        for r in rng.sample(rows, min(25, len(rows))):
            relation_items.append(
                {
                    "stratum": stratum,
                    "relation": r,
                    "source_article": articles[r["source_id"]],
                    "target_article": articles[r["target_id"]],
                    "reviewer": None,
                    "supported": None,
                    "reason": None,
                }
            )
    save(
        output / "relation_review.json",
        {
            "status": "pending_independent_review",
            "seed": seed,
            "population": {k: len(v) for k, v in strata.items()},
            "items": relation_items,
            "warning": (
                "Stratified sample; unweighted pooled precision is not population precision."
            ),
        },
    )
    # Families join identical law numbers, cross-document graph references and
    # co-gold laws. Known diagnostic families are excluded from candidate holdout.
    parents = {law: law for law, _ in corpus.values()}

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for r in relations:
        union(
            documents[articles[r["source_id"]]["document_id"]]["number"],
            documents[articles[r["target_id"]]["document_id"]]["number"],
        )
    with train_path.open(newline="", encoding="utf-8-sig") as handle:
        natural = list(csv.DictReader(handle))
    for row in natural:
        row["gold"] = re.findall(r"\d+", row["relevant_laws"])
        laws = [corpus[a][0] for a in row["gold"]]
        for law in laws[1:]:
            union(laws[0], law)
    for row in diagnostics:
        laws = [corpus[a][0] for a in row["gold"]]
        for law in laws[1:]:
            union(laws[0], law)
    excluded = {find(corpus[a][0]) for r in diagnostics for a in r["gold"]}
    qids = {r["qid"] for r in diagnostics}
    questions = {r["question"].strip() for r in diagnostics}
    candidates = []
    for row in natural:
        if row["qid"] in qids or row["question"].strip() in questions or len(row["gold"]) < 2:
            continue
        family = find(corpus[row["gold"][0]][0])
        if family in excluded:
            continue
        split = (
            "test_reserved"
            if int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), 16) % 2
            else "development"
        )
        candidates.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "gold": row["gold"],
                "source": "provided_train_csv; existing_natural_question",
                "family": family,
                "split": split,
                "mapped_gold": [a for a in row["gold"] if a in inverse],
                "gold_provisions": [
                    {
                        "aid": a,
                        "law": corpus[a][0],
                        "text": corpus[a][1],
                        "snapshot_article": inverse.get(a),
                    }
                    for a in row["gold"]
                ],
                "reviewer": None,
                "requires_linked_provisions": None,
                "category": None,
                "reviewed_answer": None,
                "review_status": "pending",
            }
        )
    candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['qid']}".encode()).hexdigest())
    # Selection does not consult graph paths, mapping coverage or retrieval results.
    chosen = [
        r
        for split in ["development", "test_reserved"]
        for r in [x for x in candidates if x["split"] == split][:30]
    ]
    save(
        output / "relational_candidates.json",
        {
            "status": "frozen_candidate_split; not_a_validated_benchmark",
            "source_sha256": sha(train_path),
            "seed": seed,
            "family_definition": (
                "Connected law-number components of co-gold laws and snapshot citations"
            ),
            "limitations": (
                "Historical training exposure unknown; reserved means not evaluated in this run."
            ),
            "excluded_diagnostic_families": sorted(excluded),
            "eligible_candidates": len(candidates),
            "items": chosen,
        },
    )
    save(
        output / "manifest.json",
        {
            "snapshot_sha256": coverage["snapshot_sha256"],
            "corpus_sha256": sha(corpus_path),
            "train_sha256": sha(train_path),
            "script_sha256": sha(__file__),
            "files": {p.name: sha(p) for p in sorted(output.glob("*.json"))},
        },
    )
    print(
        json.dumps(
            {
                "answer_questions": len(selected),
                "relations_to_review": len(relation_items),
                "natural_candidates": len(chosen),
                "eligible_natural_candidates": len(candidates),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "train", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    prepare(a.snapshot, a.corpus, a.train, a.output)
