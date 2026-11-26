"""Explain original chunk-based versus full-body identity coverage without relabeling."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from repair_coverage import body_text, load_corpus


def audit(snapshot, previous_snapshot, manifest_path, corpus_path, output):
    current = json.loads((snapshot / "coverage.json").read_text())
    previous = json.loads((previous_snapshot / "coverage.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    corpus = load_corpus(corpus_path)
    original_map = {v: k for k, v in manifest["original_ids"].items()}
    inverse = {v: k for k, v in current["original_ids"].items()}
    db = sqlite3.connect(f"file:{snapshot / 'snapshot.sqlite'}?mode=ro", uri=True)
    docs = {did: json.loads(p) for did, p in db.execute("SELECT id,payload FROM documents")}
    by_law = defaultdict(list)
    for aid, did, text in db.execute("SELECT id,document_id,text FROM articles"):
        by_law[docs[did]["number"]].append((aid, body_text(text)))
    review = []
    for sample in current["diagnostics"]:
        entries = []
        for aid in sample["gold"]:
            law, body = corpus[aid]
            entry = {
                "gold_id": aid,
                "law": law,
                "original_chunk_identity": original_map.get(aid),
                "current_full_body_identity": inverse.get(aid),
            }
            if aid not in inverse:
                entry["gap"] = sample["snapshot_unmapped_gold"][aid]
                candidates = by_law[law]
                # Diagnostic suggestions only; accepted identities are never changed.
                ranked = sorted(
                    candidates,
                    key=lambda item: difflib.SequenceMatcher(
                        None, body, item[1], autojunk=True
                    ).ratio(),
                    reverse=True,
                )
                if ranked:
                    candidate, text = ranked[0]
                    matcher = difflib.SequenceMatcher(None, body, text, autojunk=True)
                    differences = []
                    for tag, i, j, k, end in matcher.get_opcodes():
                        if tag != "equal":
                            differences.append(
                                {"operation": tag, "corpus": body[i:j], "source": text[k:end]}
                            )
                    entry["unaccepted_suggestion"] = {
                        "article_id": candidate,
                        "sequence_similarity": matcher.ratio(),
                        "corpus_chars": len(body),
                        "source_chars": len(text),
                        "differences": differences,
                    }
            entries.append(entry)
        review.append(
            {
                "qid": sample["qid"],
                "old_fully_indexed": sample["old_fully_indexed"],
                "current_fully_mapped": not sample["snapshot_unmapped_gold"],
                "gold": entries,
            }
        )
    a = set(previous["original_ids"].values())
    b = set(current["original_ids"].values())
    gaps = json.loads((snapshot / "mapping_gaps.json").read_text())
    report = {
        "status": "identity_audit; no_similarity_suggestion_accepted",
        "snapshot_sha256": current["snapshot_sha256"],
        "original_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_fingerprints_match": current["document_hashes"]
        == manifest["current_document_hashes"],
        "sentence_spacing_fix": {
            "newly_mapped_original_ids": len(b - a),
            "newly_rejected_original_ids": len(a - b),
            "rejection_reasons": dict(Counter(gaps[aid]["reason"] for aid in a - b)),
            "new_ids": sorted(b - a),
            "rejected_ids": sorted(a - b),
        },
        "coverage_transitions": dict(
            Counter(f"old_{r['old_fully_indexed']}_new_{r['current_fully_mapped']}" for r in review)
        ),
        "queries": review,
        "interpretation": (
            "Original mapping required unique stored chunk containment. Current mapping "
            "requires unique full corpus body containment and rejects many-to-one IDs. "
            "These are different identity standards, not a measured retrieval regression."
        ),
    }
    with output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["coverage_transitions"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("snapshot", "previous-snapshot", "manifest", "corpus", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    a = p.parse_args()
    audit(a.snapshot, a.previous_snapshot, a.manifest, a.corpus, a.output)
