"""Replay typed relations on a shared indexed citation-following task."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from run_reference_experiment import (
    CANDIDATES,
    COLLECTION,
    EMBEDDING_MODEL,
    SNAPSHOT,
    Snapshot,
    key,
    ordered,
    pack,
    sha,
)
from run_relation_model_comparison import KINDS, SEED, compact, read, source_patterns

from legal_graphrag.config import Settings
from legal_graphrag.llm import OllamaClient
from legal_graphrag.models import QueryFilters
from legal_graphrag.openrouter import atomic_json
from legal_graphrag.qdrant_store import QdrantStore

RULE_SNAPSHOT = Path("reports/runs/linux_20261118/coverage_v2/snapshot.sqlite")
MODES = ("hybrid", "rule", "gemma", "luna", "exact_key")
GEMMA = "gemma_by_gemma"
LUNA = "luna_by_luna"
FOLLOW = 12
ACTION = {
    "AMENDS": "sửa đổi",
    "ADDS_TO": "bổ sung",
    "REPEALS": "bãi bỏ",
    "REPLACES": "thay thế",
    "DETAILS": "quy định chi tiết",
}


def freeze(run, output):
    """Fix every target locator, question and method before running retrieval."""
    bench = read(run / "benchmark.json")
    tasks = []
    for item in bench["items"]:
        if item["split"] != "test" or item["label"] not in KINDS:
            continue
        source = item["a"]
        tasks.append(
            {
                "id": item["id"],
                "cluster": item["cluster"],
                "source_id": item["a_id"],
                "kind": item["label"],
                "source_offset": item["source_offset"],
                "passage": source["passage"],
                "query": (
                    f"Điều {source['article']} của văn bản số {source['law']} "
                    f"{ACTION[item['label']]} điều nào trong văn bản khác tại đoạn được trích?\n"
                    f"{source['passage']}"
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "benchmark_sha256": sha(run / "benchmark.json"),
        "code_sha256": sha(Path(__file__)),
        "base_rule_snapshot_sha256": sha(RULE_SNAPSHOT),
        "index_snapshot_sha256": sha(SNAPSHOT / "snapshot.sqlite"),
        "collection": COLLECTION,
        "primary": "both articles in top ten",
        "pool_limit": CANDIDATES,
        "follow_limit": FOLLOW,
        "top_k": 10,
        "context_chars": 18000,
        "tasks": sorted(tasks, key=lambda r: r["id"]),
        "scoring": "conditional replay with anchored source and same passage in all variants",
    }
    atomic_json(output / "protocol.json", protocol)
    print(f"Frozen {len(tasks)} typed retrieval tasks", flush=True)


def edge_lookup(scored, arm):
    out = defaultdict(list)
    for row in scored:
        pred = row[arm]
        if pred["label"] in KINDS:
            out[(row["a_id"], pred["label"])].append(
                (row["b_id"], pred["quote"], row["a"]["passage"])
            )
    return out


def task_edges(tasks, source, rule, store, embedder, scored):
    base = defaultdict(list)
    for (row,) in rule.db.execute("SELECT payload FROM relations ORDER BY source_id,target_id"):
        relation = json.loads(row)
        if relation["kind"] == "CITES":
            base[relation["source_id"]].append(relation)
    patterns = source_patterns()
    graphs = {"gemma": edge_lookup(scored, GEMMA), "luna": edge_lookup(scored, LUNA)}
    gold = {r["id"]: r["b_id"] for r in scored}
    vectors = embedder.embed([task["query"] for task in tasks])

    def confirm(ids, vector):
        cids = [source.chunks[aid][0] for aid in ids if aid in source.chunks]
        return ordered(
            store.evidence_hits(list(dict.fromkeys(cids)), vector, filters=QueryFilters())
        )

    for task, vector in zip(tasks, vectors, strict=True):
        aid, kind = task["source_id"], task["kind"]
        target = gold[task["id"]]
        hybrid = ordered(
            store.hybrid_query(
                task["query"],
                vector,
                filters=QueryFilters(),
                limit=CANDIDATES,
                candidate_limit=CANDIDATES,
            )
        )
        anchors = store.evidence_hits(source.chunks[aid], vector, filters=QueryFilters())
        if not anchors:
            raise ValueError(f"Missing indexed anchor {aid}")
        anchor = anchors[:1]
        tail = [h for h in hybrid if h.article_id != aid]
        scope = task["passage"]
        candidates = [r["target_id"] for r in base[aid] if compact(r["evidence"]) in scope]
        rule_hits = confirm(candidates, vector)[:FOLLOW]
        head = {"hybrid": [], "rule": rule_hits}

        for mode, graph in graphs.items():
            typed_ids = [
                b for b, quote, excerpt in graph[(aid, kind)] if excerpt == scope and quote in scope
            ]
            head[mode] = ordered(confirm(typed_ids, vector) + rule_hits, FOLLOW)
        text = unicodedata.normalize("NFC", source.article(aid))
        match = patterns[kind].match(text, task["source_offset"])
        exact_target = source.lookup.get(key(match["law"], match["article"])) if match else None
        head["exact_key"] = confirm([exact_target], vector) if exact_target else []

        modes = {}
        for mode in MODES:
            path_ids = [h.article_id for h in head[mode]]
            selected = pack(ordered(anchor + head[mode] + tail))
            retrieved = [h.article_id for h in selected]
            modes[mode] = {
                "retrieved": retrieved,
                "path_targets": path_ids,
                "both": int(aid in retrieved and target in retrieved),
                "target_at_3": int(target in retrieved[:3]),
                "gold_from_path": target in path_ids,
            }
        yield {
            "id": task["id"],
            "cluster": task["cluster"],
            "source": aid,
            "target": target,
            "kind": kind,
            "modes": modes,
        }


def paired(rows, control, treatment):
    groups = defaultdict(list)
    for row in rows:
        modes = row["modes"]
        groups[row["cluster"]].append(modes[treatment]["both"] - modes[control]["both"])
    vals = [(sum(deltas), len(deltas)) for deltas in groups.values()]
    import random

    rng = random.Random(SEED)
    draws = []
    for _ in range(5000):
        draw = rng.choices(vals, k=len(vals))
        draws.append(sum(delta for delta, _ in draw) / sum(n for _, n in draw))
    draws.sort()
    diff = sum(delta for delta, _ in vals) / sum(n for _, n in vals)
    return {"difference": diff, "ci_95": [draws[125], draws[4874]], "clusters": len(vals)}


def evaluate(run, output):
    protocol = read(output / "protocol.json")
    if (
        protocol["benchmark_sha256"] != sha(run / "benchmark.json")
        or protocol["code_sha256"] != sha(Path(__file__))
        or protocol["base_rule_snapshot_sha256"] != sha(RULE_SNAPSHOT)
        or protocol["index_snapshot_sha256"] != sha(SNAPSHOT / "snapshot.sqlite")
    ):
        raise ValueError("Frozen retrieval protocol changed")
    summary_file = output / "summary.json"
    if summary_file.exists():
        raise ValueError("Use a fresh output for rerunning retrieval")
    source = Snapshot(SNAPSHOT / "snapshot.sqlite")
    rule = Snapshot(RULE_SNAPSHOT)
    settings = replace(
        Settings.from_env(),
        qdrant_collection=COLLECTION,
        qdrant_hybrid=True,
        ollama_embed_model=EMBEDDING_MODEL,
    )
    store = QdrantStore(settings)
    store.article_mapping = source.mapping
    if len(source.mapping) != store.count():
        raise ValueError("Qdrant index does not match frozen mapping")
    embedder = OllamaClient(settings)
    candidates = read(run / "scored_test.json")
    rows = list(task_edges(protocol["tasks"], source, rule, store, embedder, candidates))
    source.db.close()
    rule.db.close()
    store.client.close()
    summary = {
        "questions": len(rows),
        "prediction_sha256": sha(run / "scored_test.json"),
        "modes": {
            mode: {
                "both": sum(r["modes"][mode]["both"] for r in rows),
                "target_at_3": sum(r["modes"][mode]["target_at_3"] for r in rows),
                "gold_from_path": sum(r["modes"][mode]["gold_from_path"] for r in rows),
            }
            for mode in MODES
        },
        "luna_minus_gemma": paired(rows, "gemma", "luna"),
        "luna_minus_rule": paired(rows, "rule", "luna"),
        "gemma_minus_rule": paired(rows, "rule", "gemma"),
    }
    atomic_json(output / "rows.json", rows)
    atomic_json(summary_file, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        freeze(args.run, args.output)
    else:
        evaluate(args.run, args.output)


if __name__ == "__main__":
    main()
