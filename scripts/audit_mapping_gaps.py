"""Produce review suggestions for unmapped diagnostic gold; never accept by similarity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from repair_coverage import body_text, load_corpus
from sklearn.feature_extraction.text import TfidfVectorizer


def audit(snapshot, corpus_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output path")
    snapshot = Path(snapshot)
    coverage = json.loads((snapshot / "coverage.json").read_text())
    corpus = load_corpus(corpus_path)
    if (
        hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()
        != coverage["inputs"]["corpus"]["sha256"]
    ):
        raise ValueError("Corpus hash mismatch")
    db = sqlite3.connect(f"file:{snapshot / 'snapshot.sqlite'}?mode=ro", uri=True)
    documents = {
        did: json.loads(payload)["number"]
        for did, payload in db.execute("SELECT id, payload FROM documents")
    }
    by_law = {}
    for aid, did, text in db.execute("SELECT id, document_id, text FROM articles"):
        by_law.setdefault(documents[did], []).append((aid, body_text(text)))
    db.close()
    rows = []
    gold = {a for row in coverage["diagnostics"] for a in row["snapshot_unmapped_gold"]}
    for aid in sorted(gold):
        law, text = corpus[aid]
        candidates = by_law.get(law, [])
        entry = {
            "original_id": aid,
            "law": law,
            "candidate_count": len(candidates),
            "accepted": False,
            "review_status": "pending",
            "suggestions": [],
        }
        if candidates:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=30000)
            matrix = vectorizer.fit_transform([text for _, text in candidates])
            scores = (matrix @ vectorizer.transform([text]).T).toarray().ravel()
            for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i][0]))[
                :3
            ]:
                current, body = candidates[i]
                entry["suggestions"].append(
                    {
                        "article_id": current,
                        "character_tfidf_similarity": float(scores[i]),
                        "corpus_excerpt": text[:350],
                        "source_excerpt": body[:350],
                    }
                )
        rows.append(entry)
    result = {
        "warning": (
            "Diagnostic similarity suggestions only; not validated identities "
            "and never used for retrieval or coverage metrics."
        ),
        "snapshot_sha256": coverage["snapshot_sha256"],
        "corpus_sha256": coverage["inputs"]["corpus"]["sha256"],
        "items": rows,
    }
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "review_items": len(rows),
                "no_source_articles": sum(not r["candidate_count"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--corpus", default="../data/corpus.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.snapshot, args.corpus, args.output)
