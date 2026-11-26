"""Check request cache isolation for reproducible answer comparisons."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_answer_experiment import cached_request


class AnswerCacheTests(unittest.TestCase):
    def test_identical_context_reuses_answer_but_model_and_prompt_changes_do_not(self):
        client = Mock()
        client._request.return_value = {"message": {"content": "saved answer"}}
        request = {"model": "generator", "messages": [{"role": "user", "content": "source A"}]}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            key, first = cached_request(client, request, "digest-a", directory)
            repeat_key, second = cached_request(client, request, "digest-a", directory)
            self.assertEqual((key, first), (repeat_key, second))
            self.assertEqual(client._request.call_count, 1)
            new_model_key, _ = cached_request(client, request, "digest-b", directory)
            new_prompt_key, _ = cached_request(
                client, dict(request, messages=[]), "digest-a", directory
            )
            self.assertEqual(len({key, new_model_key, new_prompt_key}), 3)
            self.assertEqual(client._request.call_count, 3)

    def test_failed_request_never_creates_successful_cache(self):
        client = Mock()
        client._request.side_effect = RuntimeError("service unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                cached_request(client, {"model": "generator"}, "digest", Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])
