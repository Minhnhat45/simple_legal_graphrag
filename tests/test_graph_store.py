from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_pipeline import FakeOllama, FakeVectors

from legal_graphrag.config import Settings
from legal_graphrag.graph_store import GraphStore
from legal_graphrag.pipeline import Ingestor


class GraphStoreTests(unittest.TestCase):
    def test_reads_use_configured_database_and_parameters(self) -> None:
        driver = Mock()
        driver.execute_query.return_value = ([("doc", "hash"), ("pending", None)], None, None)
        graph = GraphStore(Settings(neo4j_database="custom"), client=driver)
        self.assertEqual(graph.get_index_hashes(["doc", "pending"]), {"doc": "hash", "pending": ""})
        kwargs = driver.execute_query.call_args.kwargs
        self.assertEqual(kwargs["database_"], "custom")
        self.assertEqual(kwargs["routing_"], "r")
        self.assertEqual(kwargs["parameters_"], {"ids": ["doc", "pending"]})
        graph.close()
        driver.close.assert_called_once()

    def test_schema_errors_propagate(self) -> None:
        driver = Mock()
        driver.execute_query.side_effect = RuntimeError("permission denied")
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            GraphStore(Settings(), client=driver).ensure_schema()

    def test_environment_configuration(self) -> None:
        values = {
            "NEO4J_URI": "neo4j://example:7687",
            "NEO4J_USERNAME": "reader",
            "NEO4J_PASSWORD": "secret",
            "NEO4J_DATABASE": "legal",
        }
        with patch.dict(os.environ, values), patch("legal_graphrag.config.load_dotenv"):
            settings = Settings.from_env()
        self.assertEqual(settings.neo4j_uri, values["NEO4J_URI"])
        self.assertEqual(settings.neo4j_username, "reader")
        self.assertEqual(settings.neo4j_password, "secret")
        self.assertEqual(settings.neo4j_database, "legal")


@unittest.skipUnless(os.getenv("NEO4J_TEST_URI"), "requires a disposable Neo4j database")
class Neo4jIntegrationTests(unittest.TestCase):
    """NEO4J_TEST_URI must point to a disposable database: tests clear its contents."""

    def setUp(self) -> None:
        self.settings = Settings(
            neo4j_uri=os.environ["NEO4J_TEST_URI"],
            neo4j_username=os.getenv("NEO4J_TEST_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_TEST_PASSWORD", "legal-graphrag"),
            chunk_size_chars=500,
            chunk_overlap_chars=50,
        )
        self.graph = GraphStore(self.settings)
        self.addCleanup(self.graph.close)
        self.graph.reset()
        self.addCleanup(self.graph.reset)

    def test_ingestion_replacement_and_expansion(self) -> None:
        from legal_graphrag.csv_loader import stream_documents
        from legal_graphrag.extraction import extract_mentions

        path = Path(__file__).resolve().parents[1] / "data/sample_legal_documents.csv"
        ingestor = Ingestor(
            self.settings, graph=self.graph, vectors=FakeVectors(), ollama=FakeOllama()
        )
        first = ingestor.ingest(path)
        second = ingestor.ingest(path)
        self.assertEqual(first.documents_indexed, 3)
        self.assertEqual(second.documents_skipped, 3)
        self.assertTrue(self.graph.health())
        self.assertEqual(self.graph.stats()["chunks"], first.chunks_indexed)
        documents = list(stream_documents(path))
        relations = {
            relation
            for document in documents
            for neighbor in self.graph.expand([document.id], limit=10)
            for relation in neighbor.relations
        }
        self.assertEqual(relations, {"cites", "cited_by", "shared_metadata"})
        from legal_graphrag.models import LegalRelation

        relation = LegalRelation(
            "sample-003",
            "sample-001",
            "REPEALS",
            "sample-003:1",
            "Bãi bỏ toàn bộ Quyết định số 01/2026/QĐ-UBND",
            "test_verified",
        )
        self.graph.upsert_relations([relation])
        self.graph.upsert_relations([relation])
        self.assertEqual(self.graph.stats()["legal_relations"], 1)
        neighbors = self.graph.expand(["sample-001"], limit=10)
        self.assertEqual(neighbors[0].document_id, "sample-003")
        self.assertEqual(neighbors[0].evidence, (relation,))
        self.assertEqual(self.graph.identifier_owners()["01/2026/QĐ-UBND"], ["sample-001"])
        self.assertEqual(self.graph.relation_catalog()["sample-001"]["status"], "ready")
        self.assertIn(
            "sample-003",
            {
                d
                for d, _ in self.graph.entity_candidates(
                    "sample-001",
                    limit=10,
                )
            },
        )
        document = replace(documents[0], agency="Replacement agency")
        self.graph.prepare_documents([document], extract_mentions(document))
        self.assertEqual(self.graph.get_index_hashes([document.id]), {document.id: ""})
        self.assertEqual(self.graph.stats()["legal_relations"], 0)
        self.assertFalse(
            self.graph._read(
                "MATCH (:Document {id: $id})-[:HAS_CHUNK]->(c) RETURN c", {"id": document.id}
            )
        )
        agencies = self.graph._read(
            "MATCH (:Document {id: $id})-[:ISSUED_BY]->(a) RETURN a.name",
            {"id": document.id},
        )
        self.assertEqual([row[0] for row in agencies], ["Replacement agency"])
