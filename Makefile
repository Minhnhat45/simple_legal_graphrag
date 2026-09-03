.PHONY: install infra ingest-sample serve test lint

install:
	python -m pip install -e .

infra:
	docker compose up -d falkordb qdrant

ingest-sample:
	legal-graphrag ingest data/sample_legal_documents.csv --recreate

serve:
	legal-graphrag serve

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

lint:
	ruff check src tests

