from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .config import Settings
from .models import QueryFilters
from .service import GraphRAGService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-graphrag",
        description="Vietnamese legal GraphRAG with Neo4j, Qdrant, and Ollama",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Stream a legal-document CSV into both stores")
    ingest.add_argument("csv_path")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--collection")
    ingest.add_argument("--hybrid", action="store_true")
    ingest.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the configured graph contents and Qdrant collection before ingesting",
    )

    relations = subparsers.add_parser("build-relations", help="Build evidence-backed relations")
    relations.add_argument("csv_path")
    relations.add_argument("--limit", type=int, default=None)
    relations.add_argument("--collection")

    for name in ("build-structure", "build-semantic"):
        command = subparsers.add_parser(name, help="Extend ready indexed documents")
        command.add_argument("dataset")
        command.add_argument("--collection")
        if name == "build-semantic":
            command.add_argument("--source-limit", type=int, default=100)
            command.add_argument("--pair-limit", type=int, default=100)
            command.add_argument("--cache-dir", default=".cache/legal_relations")
            command.add_argument("--confidence", type=float, default=0.8)

    articles = subparsers.add_parser("retrieve-articles", help="Hybrid + Article graph retrieval")
    articles.add_argument("question")
    articles.add_argument("--mode", choices=["M0", "M1", "M2", "M3", "M4"], default="M4")
    articles.add_argument("--as-of", default="")
    evaluation = subparsers.add_parser("evaluate", help="Compare M0..M4 on original gold Articles")
    evaluation.add_argument("eval_csv")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--mapping", help="JSON article-ID to original gold-ID mapping")
    for command in (articles, evaluation):
        command.add_argument("--collection")
        command.add_argument("--top-k", type=int, default=10)
        command.add_argument("--config", help="JSON ExpansionConfig, including relation weights")

    ask = subparsers.add_parser("ask", help="Ask a grounded GraphRAG question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--collection")
    ask.add_argument("--mode", choices=["legacy", "M0", "M1", "M2", "M3", "M4"], default="legacy")
    ask.add_argument("--document-type", default="")
    ask.add_argument("--sector", default="")
    ask.add_argument("--field", default="")
    ask.add_argument("--status", default="")
    ask.add_argument("--agency", default="")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("stats", help="Show graph and vector-store counts")
    subparsers.add_parser("health", help="Check Neo4j, Qdrant, and Ollama")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    if getattr(args, "collection", None):
        from dataclasses import replace

        settings = replace(settings, qdrant_collection=args.collection)

    if getattr(args, "hybrid", False):
        from dataclasses import replace

        settings = replace(settings, qdrant_hybrid=True)
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "legal_graphrag.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    service = GraphRAGService.from_settings(settings)
    try:
        if args.command in {"build-structure", "build-semantic"}:
            from .csv_loader import stream_dataset
            from .pipeline import Ingestor
            from .semantic import ProvisionRelationBuilder

            documents = list(stream_dataset(args.dataset))
            catalog = service.graph.relation_catalog()
            ingestor = Ingestor(
                settings, graph=service.graph, vectors=service.vectors, ollama=service.ollama
            )
            bad = [
                d.id
                for d in documents
                if catalog.get(d.id, {}).get("status") != "ready"
                or catalog[d.id]["index_hash"] != ingestor._index_hash(d)
            ]
            if bad:
                raise ValueError(f"Ingest matching documents first: {bad[:20]}")
            chunks = [c for c, _ in service.vectors.read_document_chunks([d.id for d in documents])]
            from .chunking import chunk_document

            expected = {
                c.id: c
                for d in documents
                for c in chunk_document(
                    d, max_chars=settings.chunk_size_chars, overlap=settings.chunk_overlap_chars
                )
            }
            if {c.id: c for c in chunks} != expected:
                raise ValueError("Qdrant chunks differ from current dataset; ingest matching data")
            service.graph.ensure_schema()
            for doc in documents:
                service.graph.upsert_structure(doc, [c for c in chunks if c.document_id == doc.id])
            mapping = service.graph.chunk_article_mapping([d.id for d in documents])
            if args.command == "build-structure":
                output = {"documents": len(documents), "mapped_chunks": len(mapping)}
            else:
                output = ProvisionRelationBuilder(
                    settings, graph=service.graph, vectors=service.vectors, ollama=service.ollama
                ).build(
                    documents,
                    chunks,
                    mapping,
                    source_limit=args.source_limit,
                    pair_limit=args.pair_limit,
                    cache_dir=args.cache_dir,
                    threshold=args.confidence,
                    progress=lambda message: print(message, file=sys.stderr),
                )
        elif args.command in {"retrieve-articles", "evaluate"}:
            from dataclasses import asdict, replace
            from pathlib import Path

            from .evaluation import evaluate, load_eval, save_report
            from .ontology import ExpansionConfig
            from .retrieval import HybridRetriever

            config = (
                ExpansionConfig(**json.loads(Path(args.config).read_text()))
                if args.config
                else ExpansionConfig()
            )
            service.vectors.article_mapping = service.graph.chunk_article_mapping()
            retriever = HybridRetriever(
                settings, graph=service.graph, vectors=service.vectors, ollama=service.ollama
            )
            if args.command == "retrieve-articles":
                bundle = retriever.retrieve_articles(
                    args.question,
                    config=replace(config, mode=args.mode),
                    top_k=args.top_k,
                    as_of=args.as_of,
                )
                output = {"hits": [asdict(h) for h in bundle.hits], "trace": retriever.last_trace}
            else:
                if Path(args.output).exists():
                    raise ValueError("Output exists; choose a new experiment path")
                original_ids = json.loads(Path(args.mapping).read_text()) if args.mapping else {}
                output = evaluate(
                    retriever,
                    load_eval(args.eval_csv),
                    top_k=args.top_k,
                    config=config,
                    original_ids=original_ids,
                )
                save_report(output, args.output, eval_path=args.eval_csv)
                output = {"output": args.output, "overall": output["overall"]}
        elif args.command == "ingest":
            output = service.ingest_csv(
                args.csv_path,
                limit=args.limit,
                recreate=args.recreate,
                progress=lambda message: print(message, file=sys.stderr),
            )
        elif args.command == "build-relations":
            from .relation_builder import RelationBuilder

            output = RelationBuilder(
                settings,
                graph=service.graph,
                vectors=service.vectors,
                ollama=service.ollama,
            ).build(
                args.csv_path,
                limit=args.limit,
                progress=lambda message: print(message, file=sys.stderr),
            )
        elif args.command == "ask":
            filters = QueryFilters(
                document_type=args.document_type,
                sector=args.sector,
                field=args.field,
                status=args.status,
                agency=args.agency,
            )
            from .ontology import ExpansionConfig

            graph_config = None if args.mode == "legacy" else ExpansionConfig(mode=args.mode)
            output = service.answer(
                args.question, filters=filters, top_k=args.top_k, graph_config=graph_config
            ).to_dict()
        elif args.command == "stats":
            output = service.stats()
        elif args.command == "health":
            output = service.health()
        else:
            raise AssertionError(f"Unhandled command: {args.command}")

        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    finally:
        service.graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
