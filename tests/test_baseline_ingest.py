from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qdrant_client import QdrantClient, models

from ingest import ingest, main
from legal_graphrag.config import Settings

SAMPLE = Path(__file__).resolve().parents[1] / "data/sample_legal_documents.csv"


class BaselineIngestTests(unittest.TestCase):
    def test_cli_closes_client_on_success_and_failure(self) -> None:
        for error in (None, RuntimeError("embedding failed")):
            with self.subTest(error=error):
                # A plain Mock deliberately does not implement the context manager protocol.
                client = Mock(spec=["close"])
                with (
                    patch("sys.argv", ["ingest.py", str(SAMPLE)]),
                    patch.dict("sys.modules", {"fastembed": SimpleNamespace(
                        SparseTextEmbedding=Mock()
                    )}),
                    patch("ingest.Settings.from_env", return_value=Settings()),
                    patch("ingest.QdrantClient", return_value=client),
                    patch("ingest.ingest", side_effect=error, return_value={
                        "documents": 3, "chunks": 9
                    }) as run,
                    redirect_stdout(StringIO()),
                ):
                    if error is None:
                        main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                            main()
                    self.assertIs(run.call_args.kwargs["client"], client)
                    client.close.assert_called_once_with()

    def setUp(self) -> None:
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.embedder = Mock()
        self.embedder.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        self.bm25 = Mock()
        self.bm25.embed.side_effect = lambda texts: [
            SimpleNamespace(indices=np.array([7]), values=np.array([1.5])) for _ in texts
        ]

    def run_ingest(self, **kwargs):
        with redirect_stdout(StringIO()):
            return ingest(
                SAMPLE, Settings(embed_batch_size=2), collection="baseline",
                client=self.client, embedder=self.embedder, bm25=self.bm25, **kwargs,
            )

    def test_dense_sparse_payload_and_rebuild(self) -> None:
        counts = self.run_ingest()
        self.assertEqual(counts, {"documents": 3, "chunks": 9})
        info = self.client.get_collection("baseline")
        self.assertEqual(info.config.params.sparse_vectors["bm25"].modifier, models.Modifier.IDF)
        for name, query in (
            ("dense", [1.0, 0.0]),
            ("bm25", models.SparseVector(indices=[7], values=[1.0])),
        ):
            points = self.client.query_points("baseline", query=query, using=name).points
            self.assertTrue(points)
            self.assertIn("text", points[0].payload)
            self.assertIn("document_number", points[0].payload)
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_ingest()
        counts = self.run_ingest(limit=1, recreate=True)
        self.assertEqual(self.client.count("baseline").count, counts["chunks"])
        self.assertEqual(counts["documents"], 1)

    def test_failed_embeddings_preserve_existing_collection(self) -> None:
        self.run_ingest()
        self.embedder.embed.return_value = []
        self.embedder.embed.side_effect = None
        with self.assertRaisesRegex(ValueError, "count"):
            self.run_ingest(recreate=True)
        self.assertEqual(self.client.count("baseline").count, 9)

    def test_graph_collection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate"):
            ingest(
                SAMPLE, Settings(), collection="vn_legal_chunks", recreate=True,
                client=self.client, embedder=self.embedder, bm25=self.bm25,
            )
        self.assertFalse(self.client.collection_exists("vn_legal_chunks"))
