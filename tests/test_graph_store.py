from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from test_pipeline import SAMPLE, FakeOllama, FakeVectors

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

        path = SAMPLE
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


@unittest.skipUnless(os.getenv("NEO4J_ADDITIVE_TEST_URI"), "requires opt-in additive Neo4j test")
class ProvisionIntegrationTests(unittest.TestCase):
    """Uses unique test IDs and cleans only those IDs; never resets a database."""

    def test_provision_paths_idempotence_direction_time_and_invalidation(self):
        import uuid

        from test_extraction import make_document

        from legal_graphrag.chunking import chunk_document
        from legal_graphrag.models import ProvisionRelation
        from legal_graphrag.ontology import ExpansionConfig

        settings = replace(Settings.from_env(), neo4j_uri=os.environ["NEO4J_ADDITIVE_TEST_URI"])
        graph = GraphStore(settings)
        self.addCleanup(graph.close)
        namespace = "integration:" + uuid.uuid4().hex + ":"
        docs = [
            replace(
                make_document("Điều 1. Phải có giấy phép. Điều 2. Trách nhiệm."),
                id=namespace + str(i),
                number=namespace + str(i),
                agency="",
                sector="",
                field="",
                signer="",
            )
            for i in range(3)
        ]

        def cleanup():
            graph._query(
                "MATCH (n) WHERE (n:Document OR n:Provision OR n:Chunk) "
                "AND n.id STARTS WITH $prefix DETACH DELETE n",
                {"prefix": namespace},
            )
            graph._query(
                "MATCH (i:Instrument) WHERE i.number STARTS WITH $prefix "
                "AND NOT EXISTS { MATCH (i)--() } DELETE i",
                {"prefix": namespace},
            )

        self.addCleanup(cleanup)
        graph.ensure_schema()
        graph.prepare_documents(docs, [])
        chunks = {}
        for doc in docs:
            chunks[doc.id] = chunk_document(doc, max_chars=500, overlap=50)
            graph.upsert_chunks(chunks[doc.id])
            graph.mark_indexed(
                [{"id": doc.id, "index_hash": doc.fingerprint, "chunk_count": len(chunks[doc.id])}]
            )
            graph.upsert_structure(doc, chunks[doc.id])
        relations = [
            ProvisionRelation(
                docs[i].id + ":article:1",
                docs[i + 1].id + ":article:1",
                "SETS_CONDITION_FOR",
                chunks[docs[i].id][0].id,
                chunks[docs[i + 1].id][0].id,
                "Phải có giấy phép.",
                "Phải có giấy phép.",
                0.95,
                "{}",
                "2026-01-01",
                "2027-01-01",
                "known",
            )
            for i in range(2)
        ]
        graph.upsert_provision_relations(relations)
        graph.upsert_provision_relations(relations)
        count = graph._read(
            "MATCH (s:Article)-[r:PROVISION_RELATION]->() "
            "WHERE s.id STARTS WITH $prefix RETURN count(r)",
            {"prefix": namespace},
        )
        self.assertEqual(count[0][0], 2)
        cfg = ExpansionConfig(mode="M3", depth=2)
        result = graph.expand_articles([relations[0].source_id], config=cfg, as_of="2026-06-01")
        distant = next(r for r in result if r.article_id == relations[1].target_id)
        self.assertEqual(len(distant.path), 2)
        reverse = graph.expand_articles([relations[1].target_id], config=cfg)
        self.assertEqual(reverse[0].path[0].source_id, relations[1].source_id)
        self.assertEqual(
            graph.expand_articles([relations[0].source_id], config=cfg, as_of="2027-01-01"), []
        )
        self.assertEqual(
            graph.expand_articles([relations[0].source_id], config=replace(cfg, confidence=0.99)),
            [],
        )
        graph.prepare_documents([docs[1]], [])
        self.assertEqual(graph.expand_articles([relations[0].source_id], config=cfg), [])
