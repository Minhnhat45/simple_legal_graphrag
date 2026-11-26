from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legal_graphrag.openrouter import OpenRouterClient, OpenRouterError

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def response(content='{"ok":true}', finish="stop", model="openai/gpt-6-luna"):
    return {
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001},
    }


class OpenRouterTests(unittest.TestCase):
    def client(self, path):
        with patch.dict(os.environ, {"OPEN_ROUTER_API": "test-secret-never-serialize"}):
            return OpenRouterClient(path)

    def test_medium_reasoning_strict_schema_cache_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            with patch.object(client, "_request", return_value=response()) as request:
                result, key = client.call([{"role": "user", "content": "test"}], schema=SCHEMA)
                self.assertEqual(result, {"ok": True})
                self.assertEqual(
                    client.call([{"role": "user", "content": "test"}], schema=SCHEMA)[1], key
                )
                request.assert_called_once()
                payload = request.call_args.args[1]
                self.assertEqual(payload["reasoning"], {"effort": "medium", "exclude": True})
                self.assertEqual(payload["model"], "openai/gpt-6-luna")
                self.assertTrue(payload["response_format"]["json_schema"]["strict"])
                self.assertFalse(payload["provider"]["allow_fallbacks"])
            text = Path(tmp, f"{key}.json").read_text()
            self.assertNotIn("test-secret-never-serialize", text)
            self.assertNotIn("Authorization", text)
            self.assertEqual(client.usage()["unique_requests"], 1)

    def test_rejects_wrong_schema_truncation_refusal_and_model_substitution(self):
        for value in [
            response('{"ok":"yes"}'),
            response("prose"),
            response(finish="length"),
            response(content=None),
            response(model="other-model"),
        ]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                client = self.client(tmp)
                with (
                    patch.object(client, "_request", return_value=value),
                    self.assertRaises(OpenRouterError),
                ):
                    client.call([{"role": "user", "content": "test"}], schema=SCHEMA)
                self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)
                saved = json.loads(next(Path(tmp).glob("*.json")).read_text())
                self.assertEqual(saved["response"], value)

    def test_transport_failure_does_not_create_successful_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            with (
                patch.object(client, "_request", side_effect=OpenRouterError("HTTP 401")),
                self.assertRaises(OpenRouterError),
            ):
                client.call([{"role": "user", "content": "test"}])
            self.assertFalse(list(Path(tmp).iterdir()))
