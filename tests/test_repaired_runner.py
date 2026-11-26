"""Checkpoint guards protect immutable experiment collections on restart."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "repaired_runner", Path(__file__).resolve().parents[1] / "scripts/run_repaired_experiment.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class CheckpointTests(unittest.TestCase):
    def test_rejects_changed_provenance(self):
        identity = {
            "snapshot_sha256": "snapshot-a",
            "coverage_sha256": "coverage-a",
            "embedding_digest": "model-a",
            "collection": "new-collection",
            "code_hashes": {"retrieval.py": "code-a"},
            "embed_batch_size": 64,
            "packages": {"fastembed": "0.8.0"},
        }
        RUNNER.validate_resume({"identity": identity}, dict(identity))
        for key in identity:
            changed = dict(identity, **{key: "changed"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                RUNNER.validate_resume({"identity": identity}, changed)

    def test_checkpoint_replaces_complete_json_and_rejects_missing_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index_state.json"
            RUNNER.atomic_json(path, {"indexed": 0})
            RUNNER.atomic_json(path, {"indexed": 64})
            self.assertEqual(json.loads(path.read_text()), {"indexed": 64})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with self.assertRaisesRegex(ValueError, "existing index_state"):
                RUNNER.run("unused", "unused", Path(tmp) / "absent", resume=True)

    def test_completed_experiment_cannot_be_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "index_state.json").write_text("{}")
            Path(tmp, "evaluation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable"):
                RUNNER.run("unused", "unused", tmp, resume=True)
            with self.assertRaises(FileExistsError):
                RUNNER.run("unused", "unused", tmp)
