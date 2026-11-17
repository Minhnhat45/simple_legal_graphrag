"""Index a coverage snapshot into a NEW hybrid collection and run diagnostic controls.

Requires Qdrant and Ollama BGE-M3. The original 100 queries are diagnostics only.
No test-set claims, answer judgments, or independent review are manufactured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_graphrag.config import Settings
from legal_graphrag.evaluation import save_report, score_ranking
from legal_graphrag.experiment import SnapshotGraph, summarize_comparisons
from legal_graphrag.llm import OllamaClient
from legal_graphrag.models import DocumentChunk, LegalDocument, ProvisionRelation, QueryFilters
from legal_graphrag.ontology import ExpansionConfig
from legal_graphrag.qdrant_store import QdrantStore
from legal_graphrag.retrieval import HybridRetriever


def run(snapshot, collection, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot)
    coverage = json.loads((snapshot / "coverage.json").read_text())
    if (
        hashlib.sha256((snapshot / "snapshot.sqlite").read_bytes()).hexdigest()
        != coverage["snapshot_sha256"]
    ):
        raise ValueError("Snapshot hash mismatch")
    settings = replace(
        Settings.from_env(),
        qdrant_collection=collection,
        qdrant_hybrid=True,
        ollama_embed_model="bge-m3:latest",
        ollama_timeout_seconds=600,
    )
    vectors, ollama = QdrantStore(settings), OllamaClient(settings)
    try:
        if vectors.collection_exists():
            raise ValueError("Use a NEW collection; existing indexes are never overwritten")
        tags = ollama._request("/api/tags")["models"]
        digest = next(m["digest"] for m in tags if m["name"] == "bge-m3:latest")
        db = sqlite3.connect(f"file:{snapshot / 'snapshot.sqlite'}?mode=ro", uri=True)
        documents = {
            did: LegalDocument(**json.loads(payload))
            for did, payload in db.execute("SELECT id, payload FROM documents")
        }
        chunk_rows = list(db.execute("SELECT id, article_id, payload FROM chunks"))
        chunks = [DocumentChunk(**json.loads(payload)) for _, _, payload in chunk_rows]
        vectors.article_mapping = {cid: aid for cid, aid, _ in chunk_rows}
        article_chunks = defaultdict(list)
        for cid, aid, _ in chunk_rows:
            article_chunks[aid].append(cid)
        relations = [
            ProvisionRelation(**json.loads(p))
            for (p,) in db.execute("SELECT payload FROM relations")
        ]
        db.close()
        start = time.monotonic()
        for i in range(0, len(chunks), settings.embed_batch_size):
            batch = chunks[i : i + settings.embed_batch_size]
            embeddings = ollama.embed([c.text for c in batch])
            vectors.ensure_collection(len(embeddings[0]))
            vectors.upsert(
                batch, embeddings, {c.document_id: documents[c.document_id] for c in batch}
            )
            if i % 1600 == 0:
                print(f"indexed {i + len(batch)}/{len(chunks)}", flush=True)
        if vectors.count() != len(chunks):
            raise ValueError("Index chunk count does not match snapshot")
        index_seconds = time.monotonic() - start
        graph = SnapshotGraph(relations, article_chunks)
        shuffled = SnapshotGraph(relations, article_chunks, shuffle_seed=20261113)
        retriever = HybridRetriever(settings, graph=graph, vectors=vectors, ollama=ollama)
        base = ExpansionConfig(mode="M0", context_chars=18000)
        configurations = {
            "hybrid": base,
            "hybrid_larger": replace(base, candidate_budget=120),
            "hybrid_shared": replace(base, scoring="shared"),
            "graph_shared": replace(base, mode="M2", scoring="shared"),
            "graph_shared_path": replace(base, mode="M2", scoring="shared_path"),
            "graph_legacy": replace(base, mode="M2"),
            "shuffled_shared": replace(base, mode="M2", scoring="shared"),
        }
        originals = coverage["original_ids"]
        available = set(originals.values())
        predictions = []
        vectors._sparse_encoder()  # Warm initialization outside timed queries.
        for sample in coverage["diagnostics"]:
            query_start = time.monotonic()
            vector = ollama.embed([sample["question"]])[0]
            embedding_ms = (time.monotonic() - query_start) * 1000
            pools = {}
            for budget in (60, 120):
                start = time.monotonic()
                hits = vectors.hybrid_query(
                    sample["question"],
                    vector,
                    filters=QueryFilters(),
                    limit=budget,
                    candidate_limit=budget,
                )
                pools[budget] = hits, (time.monotonic() - start) * 1000
            baseline_ranked, baseline_candidates = set(), set()
            for mode, config in configurations.items():
                retriever.graph = shuffled if mode == "shuffled_shared" else graph
                seeds, latency = pools[config.candidate_budget]
                result = retriever.retrieve_articles(
                    sample["question"],
                    top_k=10,
                    config=config,
                    question_vector=vector,
                    hybrid_hits=seeds,
                    hybrid_latency_ms=latency,
                )
                trace = retriever.last_trace
                ranked = [originals.get(h.article_id, h.article_id) for h in result.hits]
                candidates = {originals.get(a, a) for a in trace["confirmed_candidate_articles"]}
                gold = set(sample["gold"])
                if mode == "hybrid":
                    baseline_ranked, baseline_candidates = set(ranked), candidates
                reachability = {}
                for depth in (1, 2):
                    reachable = retriever.graph.reachable(trace["seed_articles"], depth)
                    reachability[str(depth)] = {
                        "graph_node_gold": sorted(gold & {originals.get(a, a) for a in reachable}),
                        "retrievable_gold": sorted(
                            gold & {originals.get(a, a) for a in reachable if article_chunks.get(a)}
                        ),
                    }
                predictions.append(
                    {
                        "qid": sample["qid"],
                        "mode": mode,
                        "gold": sorted(gold),
                        "retrieved": ranked,
                        **score_ranking(ranked, gold, 10),
                        "candidate_recall": len(gold & candidates) / len(gold),
                        "complete_evidence": float(gold <= set(ranked)),
                        "fully_covered": gold <= available,
                        "recovered": sorted((gold & set(ranked)) - baseline_ranked),
                        "lost": sorted((gold & baseline_ranked) - set(ranked)),
                        "new_candidate_gold": sorted((gold & candidates) - baseline_candidates),
                        "latency_ms": trace["latency_ms"],
                        "embedding_ms": embedding_ms,
                        "reachability": reachability,
                        "trace": trace,
                    }
                )
            # Preserve partial predictions if a later service call fails.
            with (output / "predictions.jsonl").open("a") as handle:
                for record in predictions[-len(configurations) :]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_report(
            {
                "status": "diagnostic_retrieval_only",
                "snapshot_sha256": coverage["snapshot_sha256"],
                "coverage_sha256": hashlib.sha256(
                    (snapshot / "coverage.json").read_bytes()
                ).hexdigest(),
                "embedding_digest": digest,
                "index_seconds": index_seconds,
                "collection": collection,
                "configurations": {name: asdict(c) for name, c in configurations.items()},
                "shuffled_control": (
                    "Target endpoints permuted with seed 20261113; not legal evidence; "
                    "same expansion limits; no graph feature bonus."
                ),
                "answer_correctness": None,
                "citation_support": None,
                "independent_relational_review": "pending",
                "overall": summarize_comparisons(predictions),
                "predictions": predictions,
            },
            output / "evaluation.json",
        )
    finally:
        vectors.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.collection, args.output)
