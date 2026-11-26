"""Test article-graph expansion on natural VLSP questions with document-held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from repair_coverage import load_corpus
from run_reference_experiment import (
    CANDIDATES,
    COLLECTION,
    CONTEXT_CHARS,
    DEPENDENCY,
    EMBEDDING_MODEL,
    SAMPLE_SEED,
    SNAPSHOT,
    TOP_K,
    Snapshot,
    compact,
    display_number,
    key,
    ordered,
    sha,
)

from legal_graphrag.config import Settings
from legal_graphrag.experiment import paired_interval
from legal_graphrag.extraction import normalize_document_number
from legal_graphrag.llm import OllamaClient
from legal_graphrag.models import QueryFilters
from legal_graphrag.openrouter import atomic_json
from legal_graphrag.qdrant_store import QdrantStore

REVIEW = Path("reports/runs/linux_20261118/review/relational_candidates.json")
MAX_CHALLENGERS = 24
SEEDS = 20
TOKEN = re.compile(r"\w+")


def read(path):
    return json.loads(path.read_text())


def prepare(output, train, corpus, snapshot):
    coverage_path = snapshot / "coverage.json"
    coverage = read(coverage_path)
    if sha(corpus) != coverage["inputs"]["corpus"]["sha256"]:
        raise ValueError("Natural benchmark must use the corpus from the frozen identity mapping")
    gold_laws = {
        aid: normalize_document_number(law) for aid, (law, _) in load_corpus(corpus).items()
    }
    known = coverage["diagnostics"]
    original_split = read(REVIEW)["items"]
    development = known + [r for r in original_split if r["split"] == "development"]
    reserved = [r for r in original_split if r["split"] == "test_reserved"]
    mapped = set(coverage["original_ids"].values())
    qids = {r["qid"] for r in known + original_split}
    questions = {compact(r["question"]).casefold() for r in known + original_split}
    used_laws = {gold_laws[aid] for r in known + original_split for aid in r["gold"]}
    eligible = []
    with train.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            gold = sorted(set(re.findall(r"\d+", row["relevant_laws"])))
            if not gold or not set(gold) <= mapped or row["qid"] in qids:
                continue
            laws = {gold_laws[aid] for aid in gold}
            question = compact(row["question"])
            if question and used_laws.isdisjoint(laws) and question.casefold() not in questions:
                eligible.append((row["qid"], question, gold, laws))
    held_out = []
    # Prefer originally multi-evidence rows, then add singles from unused laws.
    for qid, question, gold, laws in sorted(
        eligible,
        key=lambda r: (len(r[2]) == 1, sha256(f"{SAMPLE_SEED}:{r[0]}".encode()).digest()),
    ):
        if used_laws.isdisjoint(laws) and question.casefold() not in questions:
            held_out.append({"qid": qid, "question": question, "gold": gold, "laws": sorted(laws)})
            used_laws.update(laws)
            questions.add(question.casefold())
    output.mkdir(parents=True, exist_ok=False)
    benchmark = {
        "inputs": {str(p): sha(p) for p in (train, corpus, coverage_path, REVIEW)},
        "selection": (
            "Natural, completely mapped train questions from unused laws. Excludes known "
            "diagnostic/development/reserved law numbers and duplicate questions. The "
            "law-disjoint sample prioritizes multi-evidence questions before single-evidence "
            "questions and never consults retrieval outputs or links."
        ),
        "known_development": [{k: r[k] for k in ("qid", "question", "gold")} for r in development],
        "legacy_reserved": [{k: r[k] for k in ("qid", "question", "gold")} for r in reserved],
        "held_out": held_out,
        "eligible_before_deduplication": len(eligible),
    }
    atomic_json(output / "benchmark.json", benchmark)
    print(
        f"Frozen {len(development)} development, {len(reserved)} reserved, "
        f"and {len(held_out)} new natural held-out questions.",
        flush=True,
    )


@dataclass(frozen=True)
class DocumentFilter(QueryFilters):
    document: str = ""

    def as_payload_map(self):
        return {**super().as_payload_map(), "document_id": self.document}


class ArticleGraph:
    def __init__(self, snapshot):
        self.links = defaultdict(dict)
        groups = defaultdict(list)
        for aid, document in snapshot.owners.items():
            number = display_number(aid)
            match = re.fullmatch(r"(\d+)([a-zđ]?)", number)
            if match and aid in snapshot.chunks:
                groups[document].append((int(match[1]), match[2], aid))
        for group in groups.values():
            ids = [item[2] for item in sorted(group)]
            for index, aid in enumerate(ids):
                for distance in range(1, 5):
                    for target in ids[max(0, index - distance) : index + distance + 1]:
                        if target != aid:
                            self.links[aid].setdefault(target, 0.62 / math.sqrt(distance))
        for a, b, payload in snapshot.db.execute(
            "SELECT source_id,target_id,payload FROM relations ORDER BY source_id,target_id,payload"
        ):
            relation = json.loads(payload)
            if relation["confidence"] >= 0.8:
                self.add(a, b, 0.89 if relation["kind"] == "CITES" else 1.0, 0.60)
        # Source-only re-extraction finds scoped cross-law references omitted by
        # the sentence-level extractor, including sentences with two numbered laws.
        self.scoped_spans = 0
        for aid, did, text in snapshot.db.execute(
            "SELECT id,document_id,text FROM articles ORDER BY id"
        ):
            if aid not in snapshot.chunks:
                continue
            for match in DEPENDENCY.finditer(text):
                target = snapshot.lookup.get(key(match["law"], match["number"]))
                if target and snapshot.owners[target] != did:
                    self.add(aid, target, 1.0, 0.66)
                    self.scoped_spans += 1
        triples = [(a, b, w) for a, edges in sorted(self.links.items()) for b, w in edges.items()]
        shuffled_targets = [b for _, b, _ in triples]
        random.Random(SAMPLE_SEED).shuffle(shuffled_targets)
        self.shuffled = defaultdict(dict)
        for (a, _, w), b in zip(triples, shuffled_targets, strict=True):
            self.shuffled[a][b] = max(w, self.shuffled[a].get(b, 0.0))

    def add(self, source, target, forward, backward):
        self.links[source][target] = max(forward, self.links[source].get(target, 0))
        self.links[target][source] = max(backward, self.links[target].get(source, 0))

    def candidates(self, seeds, links):
        support = defaultdict(float)
        for i, seed in enumerate(seeds):
            strength = 1 / math.sqrt(1 + i / 5)
            for target, weight in links[seed.article_id].items():
                support[target] = max(support[target], strength * weight)
        return support


class NaturalRetriever:
    def __init__(self, snapshot, store, embedder):
        self.snapshot = snapshot
        self.store = store
        self.embedder = embedder
        self.graph = ArticleGraph(snapshot)

    def query(self, question, vector, filters, budget):
        return self.store.hybrid_query(
            question, vector, filters=filters, limit=budget, candidate_limit=budget
        )

    def expansion(self, support, base, vector):
        novelty = [aid for aid in support if aid not in base and aid in self.snapshot.chunks]
        novelty.sort(key=lambda aid: (-support[aid], aid))
        # Some long articles have many chunks. Rank a capped disjoint proposal
        # pool by cosine, then retain only the allocated challenger slots.
        chunks = []
        for aid in novelty[:MAX_CHALLENGERS]:
            article_chunks = self.snapshot.chunks[aid]
            indices = sorted({0, len(article_chunks) // 2, len(article_chunks) - 1})
            chunks.extend(article_chunks[i] for i in indices)
        hits = self.store.evidence_hits(chunks, vector, filters=QueryFilters())
        return ordered(hits, MAX_CHALLENGERS)

    def retrieve(self, question):
        vector = self.embedder.embed([question])[0]
        base = ordered(self.query(question, vector, QueryFilters(), CANDIDATES))
        base_ranks = {hit.article_id: index for index, hit in enumerate(base, 1)}
        seeds = base[:SEEDS]
        support = {
            mode: self.graph.candidates(seeds, links)
            for mode, links in (("graph", self.graph.links), ("shuffled", self.graph.shuffled))
        }
        challengers = {
            mode: self.expansion(weights, base_ranks, vector) for mode, weights in support.items()
        }
        documents = list(dict.fromkeys(hit.document_id for hit in base[:10]))[:3]
        scopes = [
            self.query(question, vector, DocumentFilter(document=did), 8) for did in documents
        ]
        challengers["document"] = ordered(
            [
                hit
                for column in itertools.zip_longest(*scopes)
                for hit in column
                if hit is not None and hit.article_id not in base_ranks
            ],
            MAX_CHALLENGERS,
        )
        ids = {hit.chunk_id for items in [base, *challengers.values()] for hit in items}
        exact = {
            hit.chunk_id: hit.vector_score
            for hit in self.store.evidence_hits(sorted(ids), vector, filters=QueryFilters())
        }
        query_words = set(TOKEN.findall(question.casefold()))

        def features(hits, mode):
            rows = []
            for hit in hits:
                if hit.chunk_id in exact:
                    rows.append(
                        {
                            "aid": hit.article_id,
                            "chunk": hit.chunk_id,
                            "chars": len(hit.text),
                            "rank": base_ranks.get(hit.article_id, CANDIDATES + 1),
                            "cosine": exact[hit.chunk_id],
                            "overlap": len(query_words & set(TOKEN.findall(hit.text.casefold())))
                            / max(1, len(query_words)),
                            "path": support.get(mode, {}).get(hit.article_id, 0.0),
                        }
                    )
            return rows

        return {
            "hybrid": features(base, "hybrid"),
            **{
                mode: features(ordered(base[: CANDIDATES - len(hits)] + hits), mode)
                for mode, hits in challengers.items()
            },
        }


def select(rows, params):
    protected, rank_weight, lex_weight, path_weight = params
    prefix, suffix = rows[:protected], rows[protected:]
    if rank_weight is None:
        ranked = rows
    else:
        ranked = prefix + sorted(
            suffix,
            key=lambda row: (
                -(
                    row["cosine"]
                    + rank_weight / math.sqrt(row["rank"])
                    + lex_weight * row["overlap"]
                    + path_weight * row["path"]
                ),
                row["rank"],
                row["aid"],
            ),
        )
    out, used = [], 0
    for row in ranked:
        if used + row["chars"] <= CONTEXT_CHARS:
            out.append(row["aid"])
            used += row["chars"]
        if len(out) == TOP_K:
            break
    return out


DEFAULT = (0, None, 0, 0)
MODES = {
    "default_hybrid": "hybrid",
    "hybrid_rerank": "hybrid",
    "document_hybrid": "document",
    "shuffled_graph": "shuffled",
    "graph": "graph",
}


def score(pool, gold, originals, params):
    ranked = [originals.get(aid, aid) for aid in select(pool, params)]
    hits = set(ranked) & gold
    return len(hits) / len(gold), float(hits == gold)


def tune(rows, originals):
    empty_path = list(itertools.product((3, 6, 8), (0.12, 0.25, 0.45), (0.0, 0.10), (0.0,)))
    graph_grid = [(*params[:3], w) for params in empty_path for w in (0.0, 0.025, 0.06, 0.12)]
    params = {"default_hybrid": DEFAULT}
    for name, pool in MODES.items():
        if name in ("default_hybrid", "shuffled_graph"):
            continue
        grid = graph_grid if name == "graph" else empty_path
        if name == "hybrid_rerank":
            grid = [DEFAULT, *grid]
        eligible = [(r["pools"][pool], set(r["gold"])) for r in rows]

        def quality(candidate, eligible=eligible):
            metrics = [score(hits, gold, originals, candidate) for hits, gold in eligible]
            recall, complete = zip(*metrics, strict=True)
            return statistics.mean(recall), statistics.mean(complete)

        params[name] = max(grid, key=quality)
    params["shuffled_graph"] = params["graph"]
    return params


def report(rows, originals, params):
    by_mode = defaultdict(list)
    for row in rows:
        gold = set(row["gold"])
        for name, pool in MODES.items():
            candidates = row["pools"][pool]
            recall, complete = score(candidates, gold, originals, params[name])
            by_mode[name].append(
                {
                    "qid": row["qid"],
                    "recall": recall,
                    "complete": complete,
                    "candidate_recall": len(
                        {originals.get(r["aid"], r["aid"]) for r in candidates} & gold
                    ) / len(gold),
                }
            )

    def aggregate(records):
        return {
            k: statistics.mean(row[k] for row in records)
            for k in ("recall", "complete", "candidate_recall")
        }

    return {
        "questions": len(rows),
        "multi_gold": sum(len(row["gold"]) > 1 for row in rows),
        "modes": {name: aggregate(entries) for name, entries in by_mode.items()},
        "paired_recall": {
            name: paired_interval(
                [r["recall"] for r in entries],
                [r["recall"] for r in by_mode["graph"]],
                seed=SAMPLE_SEED,
                repetitions=10000,
            )
            for name, entries in by_mode.items()
            if name != "graph"
        },
        "rows": by_mode,
    }


def evaluate(output, stage, snapshot_path):
    coverage = read(snapshot_path / "coverage.json")
    benchmark = read(output / "benchmark.json")
    if sha(snapshot_path / "snapshot.sqlite") != coverage["snapshot_sha256"]:
        raise ValueError("Snapshot changed")
    mapping_path = snapshot_path / "coverage.json"
    if sha(mapping_path) != benchmark["inputs"][str(mapping_path)]:
        raise ValueError("Mapping differs from the frozen benchmark")
    source = Snapshot(snapshot_path / "snapshot.sqlite")
    settings = replace(
        Settings.from_env(),
        qdrant_collection=COLLECTION,
        qdrant_hybrid=True,
        ollama_embed_model=EMBEDDING_MODEL,
        ollama_timeout_seconds=120,
    )
    store = QdrantStore(settings)
    store.article_mapping = source.mapping
    if store.count() != len(source.mapping):
        raise ValueError("Collection changed")
    embedder = OllamaClient(settings)
    protocol = {
        "main_sha256": sha(Path(__file__)),
        "reference_sha256": sha(Path(__file__).with_name("run_reference_experiment.py")),
        "benchmark_sha256": sha(output / "benchmark.json"),
        "embedding_digest": next(
            x["digest"] for x in embedder._request("/api/tags")["models"]
            if x["name"] == EMBEDDING_MODEL
        ),
        "collection": COLLECTION,
        "primary_metric": "Recall@10",
        "primary_controls": [
            "default_hybrid", "hybrid_rerank", "document_hybrid", "shuffled_graph"
        ],
        "candidate_cap": CANDIDATES,
        "challenger_cap": MAX_CHALLENGERS,
        "top_k": TOP_K,
        "context_chars": CONTEXT_CHARS,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Frozen protocol changed")
    atomic_json(protocol_path, protocol)
    parameter_path = output / "ranking.json"
    params = read(parameter_path) if stage != "known_development" else None
    out = output / stage
    out.mkdir(exist_ok=True)
    if (out / "summary.json").exists():
        raise ValueError(f"{stage} already evaluated")
    retriever = NaturalRetriever(source, store, embedder)
    print(f"Graph built with {retriever.graph.scoped_spans} explicit cross-law spans.", flush=True)
    rows = []
    items = benchmark[stage]
    for i, item in enumerate(items, 1):
        path = out / f"{item['qid']}.json"
        if not path.exists():
            pools = retriever.retrieve(item["question"])
            atomic_json(path, {"qid": item["qid"], "gold": item["gold"], "pools": pools})
        rows.append(read(path))
        if i % 25 == 0 or i == len(items):
            print(f"{stage}: {i}/{len(items)}", flush=True)
    originals = coverage["original_ids"]
    if params is None:
        params = tune(rows, originals)
        atomic_json(parameter_path, params)
    summary = report(rows, originals, params)
    atomic_json(out / "summary.json", summary)
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, indent=2), flush=True)
    store.client.close()
    source.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "known_development", "held_out", "legacy_reserved")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/vlsp/train.csv"))
    parser.add_argument("--corpus", type=Path, default=Path("data/vlsp/corpus.csv"))
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output, args.train, args.corpus, args.snapshot)
    else:
        evaluate(args.output, args.stage, args.snapshot)


if __name__ == "__main__":
    main()
