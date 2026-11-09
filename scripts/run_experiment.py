"""Build a bounded semantic pilot and evaluate all five modes without altering gold IDs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_graphrag.config import Settings
from legal_graphrag.csv_loader import stream_dataset
from legal_graphrag.evaluation import evaluate, load_eval, save_report
from legal_graphrag.graph_store import GraphStore
from legal_graphrag.llm import OllamaClient
from legal_graphrag.ontology import ExpansionConfig
from legal_graphrag.qdrant_store import QdrantStore
from legal_graphrag.retrieval import HybridRetriever
from legal_graphrag.semantic import ProvisionRelationBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise ValueError("Use a new output directory; evaluations are never overwritten")
    settings = replace(
        Settings.from_env(),
        qdrant_collection=manifest["collection"],
        ollama_embed_model="bge-m3:latest",
        ollama_timeout_seconds=600,
        relation_representative_chunks=1,
    )
    graph, vectors, ollama = GraphStore(settings), QdrantStore(settings), OllamaClient(settings)
    try:
        documents = list(stream_dataset(manifest["dataset"]))
        if {d.id: d.fingerprint for d in documents} != manifest["current_document_hashes"]:
            raise ValueError("Current dataset changed since manifest; prepare a new snapshot")
        namespace = manifest["namespace"]
        mapping = graph.chunk_article_mapping()
        mapping = {cid: aid for cid, aid in mapping.items() if aid.startswith(namespace)}
        documents = [replace(d, id=namespace + d.id) for d in documents]
        chunks = [c for c, _ in vectors.read_document_chunks([d.id for d in documents])]
        vectors.article_mapping = mapping
        if not args.skip_build:
            if (output / "relations.json").exists():
                raise ValueError("Relation report already exists")
            report = ProvisionRelationBuilder(
                settings, graph=graph, vectors=vectors, ollama=ollama
            ).build(
                documents,
                chunks,
                mapping,
                source_limit=args.sources,
                pair_limit=args.pairs,
                cache_dir=output / "verifier_cache",
                progress=lambda s: print(s, flush=True),
            )
            save_report(report, output / "relations.json")
        samples = load_eval(args.eval)
        originals = manifest["original_ids"]
        available = set(originals.values())
        coverage = {
            "total_queries": len(samples),
            "fully_covered_queries": sum(set(s["gold"]) <= available for s in samples),
            "unmapped_gold_ids": sorted({a for s in samples for a in s["gold"]} - available),
        }
        retriever = HybridRetriever(settings, graph=graph, vectors=vectors, ollama=ollama)
        # Warm sparse encoder and database paths outside timed searches.
        retriever.retrieve_articles(samples[0]["question"], config=ExpansionConfig(mode="M0"))
        report = evaluate(retriever, samples, top_k=10, original_ids=originals)
        report["coverage"] = coverage
        report["manifest"] = str(Path(args.manifest).resolve())
        save_report(report, output / "evaluation.json", eval_path=args.eval)
        print(
            json.dumps({"coverage": coverage, "overall": report["overall"]}, indent=2), flush=True
        )
    finally:
        graph.close()
        vectors.client.close()


if __name__ == "__main__":
    main()
