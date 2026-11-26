"""Benchmark cross-document citation retrieval against dense+BM25 controls."""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import file_digest, sha256
from math import comb
from pathlib import Path

from legal_graphrag.config import Settings
from legal_graphrag.evaluation import score_ranking
from legal_graphrag.experiment import paired_interval
from legal_graphrag.extraction import DOCUMENT_NUMBER_RE, normalize_document_number
from legal_graphrag.llm import OllamaClient
from legal_graphrag.models import QueryFilters
from legal_graphrag.openrouter import atomic_json
from legal_graphrag.qdrant_store import QdrantStore

SAMPLE_SEED = 20261121
TOP_K = 10
CONTEXT_CHARS = 18000
CANDIDATES = 60
FOLLOW_CANDIDATES = 12
EMBEDDING_MODEL = "bge-m3:latest"
COLLECTION = "legal_openrouter_20261119_v1"
SNAPSHOT = Path("reports/runs/openrouter_20261119/snapshot")

ARTICLE_NUMBER = re.compile(r"\d+[a-zđ]?", re.I)
SOURCE_ARTICLE = re.compile(r"^Điều (\d+[a-zđ]?) của văn bản số ([^,]+),", re.I)
CUE = re.compile(r"«([^»]+)»")
DEPENDENCY = re.compile(
    r"\b(?:theo\s+(?:quy\s+định\s+)?(?:tại\s+)?|quy\s+định\s+tại\s+)"
    r"(?:(?:điểm\s+\w+|khoản\s+\d+)\s+){0,2}"
    r"(?P<article>Điều\s+(?P<number>\d+[a-zđ]?))\s+(?:của\s+)?"
    r"(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Luật|Bộ\s+luật)\s+(?:số\s+)?"
    r"(?P<law>\d+/\d{4}/[\w-]+)(?![\w/-])",
    re.I,
)
# Used at retrieval for citation text after locating the quoted action in the source.
CITATION_NUMBER = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b", re.I)
ASSIGNMENTS = ("hybrid", "hybrid_rewrite", "hybrid_scoped", "graph", "shuffled", "exact_key")


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def compact(text):
    return " ".join(text.split())


def display_number(article_id):
    return article_id.rpartition(":article:")[2].casefold()


def key(law, number):
    return normalize_document_number(law), number.casefold()


class Snapshot:
    def __init__(self, database):
        self.db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        self.documents = dict(
            self.db.execute("SELECT id, json_extract(payload, '$.number') FROM documents")
        )
        self.number_counts = Counter(normalize_document_number(n) for n in self.documents.values())
        self.chunks = defaultdict(list)
        self.mapping = {}
        for cid, aid in self.db.execute("SELECT id, article_id FROM chunks ORDER BY id"):
            self.mapping[cid] = aid
            self.chunks[aid].append(cid)
        self.owners = dict(self.db.execute("SELECT id, document_id FROM articles ORDER BY id"))
        self.lookup = {
            key(self.documents[did], display_number(aid)): aid
            for aid, did in self.owners.items()
            if self.unique_document(did)
            and ARTICLE_NUMBER.fullmatch(display_number(aid))
            and aid in self.chunks
        }

    def unique_document(self, did):
        number = normalize_document_number(self.documents[did])
        return bool(number and self.number_counts[number] == 1)

    def article(self, aid):
        return self.db.execute("SELECT text FROM articles WHERE id=?", (aid,)).fetchone()[0]

    def cross_document_citations(self):
        rows = self.db.execute(
            """
            SELECT r.payload FROM relations r
            JOIN articles a ON a.id=r.source_id JOIN articles b ON b.id=r.target_id
            WHERE a.document_id != b.document_id
            ORDER BY r.source_id, r.target_id, r.payload
            """
        )
        edges = []
        for (payload,) in rows:
            relation = json.loads(payload)
            if relation["kind"] == "CITES" and relation["method"] == "exact_article_reference":
                edges.append(relation)
        return edges


def prepare(snapshot, output):
    """Select tasks from article text using no graph table, model, or rankings."""
    source = Snapshot(snapshot / "snapshot.sqlite")
    examples = []
    for aid, did, raw in source.db.execute(
        "SELECT id, document_id, text FROM articles ORDER BY id"
    ):
        law = source.documents[did]
        if source.lookup.get(key(law, display_number(aid))) != aid:
            continue
        body = compact(raw)
        for match in DEPENDENCY.finditer(body):
            target = source.lookup.get(key(match["law"], match["number"]))
            if target is None or source.owners[target] == did:
                continue
            beginning = max(body.rfind(char, 0, match.start()) for char in (".", ";", ":")) + 1
            prefix = body[beginning : match.start()].strip()
            # Preserve an exact action cue while keeping the target identifier hidden.
            cue = " ".join(prefix.split()[-24:]).strip(" ,")
            if len(cue) < 35 or body.count(cue) != 1 or any(c in cue for c in "«»"):
                continue
            if DOCUMENT_NUMBER_RE.search(cue) or CITATION_NUMBER.search(cue):
                continue
            qid = sha256(f"{SAMPLE_SEED}:{aid}:{match.start()}".encode()).hexdigest()[:20]
            question = (
                f"Điều {display_number(aid)} của văn bản số {law}, trong đoạn «{cue}», "
                "dẫn chiếu điều nào của văn bản khác? Tìm cả điều gốc và điều được dẫn chiếu."
            )
            examples.append(
                {
                    "qid": qid,
                    "question": question,
                    "source": aid,
                    "target": target,
                    "source_law": law,
                    "target_law": source.documents[source.owners[target]],
                    "source_quote": body[beginning : match.end()].strip(),
                    "target_opening": compact(source.article(target))[:300],
                }
            )
    # Each law may appear once across all selected examples, in either endpoint role.
    selected, used = [], set()
    for row in sorted(examples, key=lambda item: item["qid"]):
        laws = {normalize_document_number(row[f"{side}_law"]) for side in ("source", "target")}
        if used.isdisjoint(laws):
            selected.append(row)
            used.update(laws)
    if len(selected) < 76:
        raise ValueError(f"Only {len(selected)} disjoint law pairs")
    for index, row in enumerate(selected):
        row["split"] = "development" if index < 25 else "held_out"
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "benchmark.json",
        {
            "snapshot_sha256": sha(snapshot / "snapshot.sqlite"),
            "builder_sha256": sha(Path(__file__)),
            "selection": (
                "Source-text explicit operative article references with two uniquely indexed "
                "documents and an unambiguous reference cue. Sorted by seeded hash before "
                "taking the first eligible pair for each unused document. The first 25 "
                "pairs are development and the rest are held out."
            ),
            "label_source": "deterministic source-derived labels; no independent human review",
            "eligible_spans": len(examples),
            "items": selected,
        },
    )
    print(f"Frozen {len(selected)} questions ({len(examples)} eligible citation spans)", flush=True)
    source.db.close()


@dataclass(frozen=True)
class DocumentNumberFilter(QueryFilters):
    law: str = ""

    def as_payload_map(self):
        return {**super().as_payload_map(), "document_number": self.law}


def ordered(hits, budget=CANDIDATES):
    values = {}
    for hit in hits:
        if hit.article_id:
            values.setdefault(hit.article_id, hit)
    return list(values.values())[:budget]


def pack(hits):
    selected, length = [], 0
    for hit in ordered(hits):
        if length + len(hit.text) <= CONTEXT_CHARS:
            selected.append(hit)
            length += len(hit.text)
            if len(selected) == TOP_K:
                break
    return selected


def citation_query(question, snapshot):
    match, cue = SOURCE_ARTICLE.match(question), CUE.search(question)
    if match is None or cue is None:
        return None
    aid = snapshot.lookup.get(key(match[2], match[1]))
    if aid is None:
        return None
    body = compact(snapshot.article(aid))
    if body.count(cue[1]) != 1:
        return None
    start = body.index(cue[1]) + len(cue[1])
    following = body[start : start + 300].lstrip(" ,")
    # Resolve the first explicit article and law number in the adjacent span.
    article, law = CITATION_NUMBER.search(following), DOCUMENT_NUMBER_RE.search(following)
    if article is None or law is None or not article.start() < law.start() < article.end() + 75:
        return None
    return aid, cue[1], article[1], law[0], following[article.start() : law.end()]


class ReferenceRetriever:
    def __init__(self, snapshot, store, embedder):
        self.snapshot = snapshot
        self.store = store
        self.embedder = embedder
        edges = snapshot.cross_document_citations()
        shuffled = [edge["target_id"] for edge in edges]
        random.Random(SAMPLE_SEED).shuffle(shuffled)
        self.adjacency = defaultdict(list)
        for edge, replacement in zip(edges, shuffled, strict=True):
            self.adjacency[edge["source_id"]].append((edge, replacement))
        self.edge_count = len(edges)

    def evidence(self, cids, vector):
        return self.store.evidence_hits(cids, vector, filters=QueryFilters())

    def retrieve(self, question):
        started = time.monotonic()
        ref = citation_query(question, self.snapshot)
        texts = [question, ref[-1]] if ref else [question]
        vector, *follow = self.embedder.embed(texts)
        first = self.store.hybrid_query(
            question, vector, filters=QueryFilters(), limit=CANDIDATES, candidate_limit=CANDIDATES
        )
        rankings = {"hybrid": ordered(first)}
        trace = {"followup_query": ref[-1] if ref else None}
        if ref is None:
            return {mode: rankings["hybrid"] for mode in ASSIGNMENTS}, trace
        aid, cue, article, law, citation = ref
        anchor = self.evidence(self.snapshot.chunks[aid], vector)
        anchor.sort(key=lambda hit: (cue in compact(hit.text), hit.vector_score), reverse=True)
        primary = anchor[:1]
        raw = self.store.hybrid_query(
            citation,
            follow[0],
            filters=QueryFilters(),
            limit=FOLLOW_CANDIDATES,
            candidate_limit=FOLLOW_CANDIDATES,
        )
        tail = ordered(first)[: CANDIDATES - FOLLOW_CANDIDATES - len(primary)]
        rankings["hybrid_rewrite"] = ordered([*primary, *raw, *tail])
        exact_target = self.snapshot.lookup.get(key(law, article))
        # Scope by the law number only. This control still relies on hybrid ranking
        # to identify the target article, including when another article quotes it.
        normalized = normalize_document_number(law)
        canonical_law = next(
            (number for number in self.snapshot.documents.values()
             if normalize_document_number(number) == normalized),
            law,
        )
        scoped = self.store.hybrid_query(
            citation,
            follow[0],
            filters=DocumentNumberFilter(law=canonical_law),
            limit=FOLLOW_CANDIDATES,
            candidate_limit=FOLLOW_CANDIDATES,
        )
        flat_backfill = ordered([*scoped, *raw], FOLLOW_CANDIDATES)
        rankings["hybrid_scoped"] = ordered([*primary, *flat_backfill, *tail])

        matching = [
            (edge, replacement)
            for edge, replacement in self.adjacency[aid]
            if compact(citation).casefold() in compact(edge["evidence"]).casefold()
            and edge["source_chunk_id"] in self.snapshot.chunks[aid]
        ]
        for mode in ("graph", "shuffled"):
            targets = {}
            for edge, replacement in matching:
                if mode == "graph":
                    targets[edge["target_id"]] = [edge["target_chunk_id"]]
                else:
                    targets[replacement] = self.snapshot.chunks[replacement][:1]
            route = self.evidence(
                [cid for cids in list(targets.values())[:FOLLOW_CANDIDATES] for cid in cids],
                follow[0],
            )
            followup = ordered([*route, *scoped, *raw], FOLLOW_CANDIDATES)
            rankings[mode] = ordered([*primary, *followup, *tail])
            trace[f"{mode}_targets"] = [hit.article_id for hit in route]
        target_hits = (
            self.evidence(self.snapshot.chunks[exact_target], follow[0]) if exact_target else []
        )
        rankings["exact_key"] = ordered(
            [*primary, *ordered([*target_hits, *scoped, *raw], FOLLOW_CANDIDATES), *tail]
        )
        trace["anchor"] = aid
        trace["all_modes_ms"] = 1000 * (time.monotonic() - started)
        return rankings, trace


def paired(baseline, treatment):
    result = paired_interval(baseline, treatment, seed=SAMPLE_SEED, repetitions=10000)
    wins = sum(b < t for b, t in zip(baseline, treatment, strict=True))
    losses = sum(b > t for b, t in zip(baseline, treatment, strict=True))
    result.update(
        wins=wins,
        losses=losses,
        ties=len(baseline) - wins - losses,
        exact_two_sided_p=min(
            1.0,
            2 * sum(comb(wins + losses, i) for i in range(min(wins, losses) + 1))
            / 2 ** (wins + losses),
        ),
    )
    return result


def evaluate(snapshot, output, split):
    db_path = snapshot / "snapshot.sqlite"
    benchmark_path = output / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark["snapshot_sha256"] != sha(db_path):
        raise ValueError("Source snapshot differs from the frozen benchmark")
    source = Snapshot(db_path)
    coverage = json.loads((snapshot / "coverage.json").read_text())
    settings = replace(
        Settings.from_env(),
        qdrant_collection=COLLECTION,
        qdrant_hybrid=True,
        ollama_embed_model=EMBEDDING_MODEL,
        ollama_timeout_seconds=120,
    )
    store = QdrantStore(settings)
    store.article_mapping = source.mapping
    if store.count() != coverage["chunks"] or store.count() != len(source.mapping):
        raise ValueError("Qdrant index count differs from the frozen snapshot")
    embedder = OllamaClient(settings)
    digest = next(
        model["digest"] for model in embedder._request("/api/tags")["models"]
        if model["name"] == EMBEDDING_MODEL
    )
    protocol = {
        "benchmark_sha256": sha(benchmark_path),
        "code_sha256": sha(Path(__file__)),
        "collection": COLLECTION,
        "embedding": digest,
        "candidate_budget": CANDIDATES,
        "followup_budget": FOLLOW_CANDIDATES,
        "top_k": TOP_K,
        "context_chars": CONTEXT_CHARS,
        "primary_metric": "both source and cited article retrieved at 10",
        "primary_controls": ["hybrid_scoped", "shuffled"],
        "comparisons": list(ASSIGNMENTS),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Run protocol changed")
    atomic_json(protocol_path, protocol)

    result_dir = output / split
    result_dir.mkdir(exist_ok=True)
    if (result_dir / "summary.json").exists():
        raise ValueError(f"Already evaluated {split}")
    retriever = ReferenceRetriever(source, store, embedder)
    samples = [item for item in benchmark["items"] if item["split"] == split]
    rows = []
    for i, item in enumerate(samples, start=1):
        record_path = result_dir / f"{item['qid']}.json"
        if record_path.exists():
            rows.append(json.loads(record_path.read_text()))
            continue
        rankings, trace = retriever.retrieve(item["question"])
        row = {"qid": item["qid"], "gold": [item["source"], item["target"]], "trace": trace}
        row["modes"] = {}
        for mode, candidates in rankings.items():
            results = pack(candidates)
            ranked = [hit.article_id for hit in results]
            row["modes"][mode] = {
                "retrieved": ranked,
                "chunk_ids": [hit.chunk_id for hit in results],
                "context_chars": sum(len(hit.text) for hit in results),
                "candidates": len(candidates),
                "recall": score_ranking(ranked, row["gold"], TOP_K)["recall"],
                "complete": int(set(row["gold"]) <= set(ranked)),
                "target": int(item["target"] in ranked),
                "target_at_3": int(item["target"] in ranked[:3]),
            }
        atomic_json(record_path, row)
        rows.append(row)
        if i % 10 == 0 or i == len(samples):
            print(f"{split}: {i}/{len(samples)}", flush=True)
    summary = {"questions": len(samples), "citation_edges": retriever.edge_count, "modes": {}}
    for mode in ASSIGNMENTS:
        summary["modes"][mode] = {
            name: statistics.mean(r["modes"][mode][name] for r in rows)
            for name in ("recall", "complete", "target", "target_at_3", "candidates")
        }
    summary["paired_complete_vs_graph"] = {
        control: paired(
            [r["modes"][control]["complete"] for r in rows],
            [r["modes"]["graph"]["complete"] for r in rows],
        )
        for control in ("hybrid", "hybrid_rewrite", "hybrid_scoped", "shuffled", "exact_key")
    }
    atomic_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    store.client.close()
    source.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=SNAPSHOT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("stage", choices=("prepare", "development", "held_out"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.snapshot, args.output)
    else:
        evaluate(args.snapshot, args.output, args.stage)


if __name__ == "__main__":
    main()
