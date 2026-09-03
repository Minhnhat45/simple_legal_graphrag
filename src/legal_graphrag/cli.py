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
        description="Vietnamese legal GraphRAG with FalkorDB, Qdrant, and Ollama",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Stream a legal-document CSV into both stores")
    ingest.add_argument("csv_path")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the configured graph contents and Qdrant collection before ingesting",
    )

    ask = subparsers.add_parser("ask", help="Ask a grounded GraphRAG question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
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
    subparsers.add_parser("health", help="Check FalkorDB, Qdrant, and Ollama")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()

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
    if args.command == "ingest":
        output = service.ingest_csv(
            args.csv_path,
            limit=args.limit,
            recreate=args.recreate,
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
        output = service.answer(args.question, filters=filters, top_k=args.top_k).to_dict()
    elif args.command == "stats":
        output = service.stats()
    elif args.command == "health":
        output = service.health()
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
