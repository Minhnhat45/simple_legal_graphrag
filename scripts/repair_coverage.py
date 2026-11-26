"""Build a new, complete local article/chunk snapshot and audit exact corpus mappings.

No vector service or old embedding is required. SQLite FTS is a searchable staging
index, NOT a replacement for the controlled BGE-M3 hybrid experiment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_graphrag.chunking import chunk_document
from legal_graphrag.csv_loader import normalize_text, stream_dataset
from legal_graphrag.semantic import explicit_article_relations
from legal_graphrag.structure import parse_structure

DICT = re.compile(
    r"\{\s*['\"]aid['\"]:\s*\d+,\s*['\"]content_Article['\"]:\s*(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\}",
    re.S,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    text = re.sub(r"\s*([\"'])\s*", r"\1", text)
    # Flattened corpus exports sometimes omit a sentence separator space
    # ("...định.Trường hợp..."). Preserve every letter, number and punctuation
    # mark; canonicalize only spacing before an uppercase sentence start.
    text = re.sub(
        r"([.;:!?])\s*([^\W\d_])",
        lambda match: match[1] + " " + match[2] if match[2].isupper() else match[0],
        text,
    )
    return normalize_text(text.replace(r"\n", " ").replace(r"\t", " ").replace(r"\r", " "))


def load_corpus(path):
    csv.field_size_limit(sys.maxsize)
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row["content"]
            if DICT.sub("", raw).strip("[] \r\n\t,"):
                raise ValueError(f"Unparsed corpus content for {row['law_id']}")
            for match in DICT.finditer(raw):
                article = ast.literal_eval(match[0])
                aid = str(article["aid"])
                if aid in result:
                    raise ValueError(f"Duplicate corpus aid: {aid}")
                result[aid] = (row["law_id"], body_text(article["content_Article"]))
    if not result:
        raise ValueError("Empty corpus")
    return result


def build(dataset, corpus_path, analysis_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    corpus = load_corpus(corpus_path)
    analysis = json.loads(Path(analysis_path).read_text())
    if sha(corpus_path) != analysis["inputs"]["corpus"]["sha256"]:
        raise ValueError("Corpus differs from the frozen failure analysis")
    documents = list(stream_dataset(dataset))
    articles, by_law, chunks, chunk_map = {}, defaultdict(list), [], {}
    no_articles = []
    for doc in documents:
        parsed = [p for p in parse_structure(doc) if p.level == "Article"]
        if not parsed:
            no_articles.append(doc.id)
        for article in parsed:
            articles[article.id] = article
            by_law[doc.number].append(article.id)
            # Chunk each parsed provision independently, including repeated headings.
            # Its explicit owner is never guessed from the display section label.
            for i, chunk in enumerate(
                chunk_document(replace(doc, full_text=article.text), max_chars=1800, overlap=250)
            ):
                chunk = replace(chunk, id=f"{article.id}:chunk:{i}")
                chunks.append(chunk)
                chunk_map[chunk.id] = article.id
    normalized = {aid: body_text(a.text) for aid, a in articles.items()}
    matches, gaps = {}, {}
    for aid, (law, body) in corpus.items():
        candidates = [a for a in by_law[law] if len(body) >= 30 and body in normalized[a]]
        if len(candidates) == 1:
            matches[aid] = candidates[0]
        else:
            gaps[aid] = {
                "law_id": law,
                "reason": "ambiguous" if candidates else "no_exact_article_match",
                "matches": candidates,
            }
    reverse = defaultdict(list)
    for aid, current in matches.items():
        reverse[current].append(aid)
    for current, aids in reverse.items():
        if len(aids) > 1:
            for aid in aids:
                gaps[aid] = {
                    "law_id": corpus[aid][0],
                    "reason": "multiple_original_ids",
                    "matches": [current],
                }
                del matches[aid]
    graph_start = time.monotonic()
    relations = explicit_article_relations(documents, chunks, chunk_map)
    graph_seconds = time.monotonic() - graph_start
    db = sqlite3.connect(output / "snapshot.sqlite")
    db.executescript("""
        CREATE TABLE documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE articles (id TEXT PRIMARY KEY, document_id TEXT,
            text TEXT NOT NULL, original_id TEXT);
        CREATE TABLE chunks (id TEXT PRIMARY KEY, article_id TEXT, payload TEXT NOT NULL);
        CREATE VIRTUAL TABLE article_search USING fts5(id UNINDEXED, text);
        CREATE TABLE relations (source_id TEXT, target_id TEXT, payload TEXT NOT NULL);
    """)
    originals = {v: k for k, v in matches.items()}
    db.executemany(
        "INSERT INTO documents VALUES (?, ?)",
        [(d.id, json.dumps(asdict(d), ensure_ascii=False)) for d in documents],
    )
    db.executemany(
        "INSERT INTO articles VALUES (?, ?, ?, ?)",
        [(a.id, a.document_id, a.text, originals.get(a.id)) for a in articles.values()],
    )
    db.executemany(
        "INSERT INTO article_search VALUES (?, ?)", [(a.id, a.text) for a in articles.values()]
    )
    db.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?)",
        [(c.id, chunk_map[c.id], json.dumps(asdict(c), ensure_ascii=False)) for c in chunks],
    )
    db.executemany(
        "INSERT INTO relations VALUES (?, ?, ?)",
        [(r.source_id, r.target_id, json.dumps(asdict(r), ensure_ascii=False)) for r in relations],
    )
    db.commit()
    assert db.execute("SELECT count(*) FROM article_search").fetchone()[0] == len(articles)
    assert not db.execute(
        "SELECT id FROM articles WHERE id NOT IN (SELECT article_id FROM chunks)"
    ).fetchall()
    db.close()
    rows = [r for r in analysis["rows"] if r["mode"] == "M0"]
    audit = []
    for row in rows:
        gold = set(row["gold_articles"])
        audit.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "gold": sorted(gold),
                "group": row["group"],
                "old_fully_indexed": row["all_gold_articles_indexed"],
                "snapshot_mapped_gold": sorted(gold & matches.keys()),
                "snapshot_unmapped_gold": {a: gaps[a] for a in sorted(gold - matches.keys())},
                "old_retrieved": row["retrieved_articles"],
                "old_candidate_gold": row["candidate_gold_articles"],
            }
        )
    report = {
        "status": "local_searchable_snapshot_only; vector_index_and_new_retrieval_not_run",
        "inputs": {
            "corpus": {"path": str(Path(corpus_path).resolve()), "sha256": sha(corpus_path)},
            "analysis": {"path": str(Path(analysis_path).resolve()), "sha256": sha(analysis_path)},
        },
        "document_hashes": {d.id: d.fingerprint for d in documents},
        "chunk_size_chars": 1800,
        "chunk_overlap_chars": 250,
        "mapping_rule": (
            "Full normalized corpus article body (>=30 chars) occurs in exactly one "
            "parsed source article of the identical law number; reject many-to-one "
            "matches. NFC, whitespace, uppercase sentence-start spacing and quote "
            "typography normalization only; "
            "source evidence remains unchanged. No ordinal inference."
        ),
        "source_hashes": {
            str(path): sha(path)
            for path in [
                Path(__file__),
                *sorted(
                    Path(__file__).resolve().parents[1].joinpath("src/legal_graphrag").glob("*.py")
                ),
            ]
        },
        "documents": len(documents),
        "articles": len(articles),
        "chunks": len(chunks),
        "documents_without_parsed_articles": no_articles,
        "corpus_articles": len(corpus),
        "mapped_original_articles": len(matches),
        "original_ids": originals,
        "gap_counts": dict(Counter(g["reason"] for g in gaps.values())),
        "resolved_citations": len(relations),
        "graph_build_seconds": graph_seconds,
        "queries": len(rows),
        "old_fully_indexed_queries": sum(r["all_gold_articles_indexed"] for r in rows),
        "snapshot_fully_mapped_queries": sum(not r["snapshot_unmapped_gold"] for r in audit),
        "snapshot_mapped_gold_occurrences": sum(len(r["snapshot_mapped_gold"]) for r in audit),
        "gold_occurrences": sum(len(r["gold"]) for r in audit),
        "build_seconds": time.monotonic() - start,
        "snapshot_sha256": sha(output / "snapshot.sqlite"),
        "diagnostics": audit,
    }
    for name, payload in (("coverage.json", report), ("mapping_gaps.json", gaps)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "document_hashes",
                    "original_ids",
                    "diagnostics",
                    "documents_without_parsed_articles",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--corpus", default="../data/corpus.csv")
    parser.add_argument("--analysis", default="reports/failure_analysis_20261107.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.dataset, args.corpus, args.analysis, args.output)
