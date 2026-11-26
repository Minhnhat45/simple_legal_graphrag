"""Audited OpenRouter chat requests with strict schemas and immutable caches.

Credentials are read from the environment and never serialized in research logs.
Reasoning effort is fixed at medium; no fallback model is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import load_dotenv


class OpenRouterError(RuntimeError):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class OpenRouterClient:
    model = "openai/gpt-6-luna"
    reasoning_effort = "medium"
    endpoint = "https://openrouter.ai/api/v1"

    def __init__(self, cache_dir, *, timeout=600):
        load_dotenv()
        self._key = os.getenv("OPEN_ROUTER_API", "")
        if not self._key:
            raise OpenRouterError("OPEN_ROUTER_API is missing")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._locks = {}
        self._guard = threading.Lock()

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not expose request headers or arbitrary provider error bodies.
            raise OpenRouterError(f"OpenRouter HTTP {exc.code} at {path}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OpenRouterError(f"OpenRouter transport error: {type(exc).__name__}") from None

    def model_metadata(self):
        model = next((m for m in self._request("/models")["data"] if m["id"] == self.model), None)
        if model is None:
            raise OpenRouterError("Requested model is unavailable; no model substitution allowed")
        required = {"reasoning", "response_format", "structured_outputs"}
        if not required <= set(model.get("supported_parameters", [])):
            raise OpenRouterError("Requested model lacks required reasoning/schema support")
        return model

    def call(self, messages, *, schema=None, max_tokens=8192):
        """Return a validated result and cache key; identical requests share one call."""
        payload = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
            "max_tokens": max_tokens,
            "stream": False,
            "provider": {"require_parameters": True, "allow_fallbacks": False},
            "seed": 20261119,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "legal_research_result", "strict": True, "schema": schema},
            }
        identity = {"endpoint": self.endpoint, "payload": payload}
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        path = self.cache_dir / f"{key}.json"
        with lock:
            if path.exists():
                record = json.loads(path.read_text())
                if record["request"] != identity:
                    raise OpenRouterError("Request cache identity mismatch")
            else:
                started = time.monotonic()
                response = self._request("/chat/completions", payload)
                record = {
                    "request": identity,
                    "response": response,
                    "wall_seconds": time.monotonic() - started,
                }
                atomic_json(path, record)
        response = record["response"]
        if response.get("error") or not response.get("choices"):
            raise OpenRouterError(f"Provider failed; inspect saved response {key}")
        returned_model = response.get("model", "")
        if returned_model != self.model and not returned_model.startswith(self.model + "-"):
            raise OpenRouterError(f"Unexpected model in saved response {key}")
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content")
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise OpenRouterError(f"Incomplete/refused model output; inspect saved response {key}")
        if schema is None:
            result = content.strip()
        else:
            import jsonschema

            try:
                result = json.loads(content)
                jsonschema.validate(result, schema)
            except (ValueError, jsonschema.ValidationError) as exc:
                raise OpenRouterError(f"Invalid structured output in saved response {key}") from exc
        return result, key

    def usage(self):
        records = [json.loads(p.read_text()) for p in sorted(self.cache_dir.glob("*.json"))]
        usage = [r["response"].get("usage", {}) for r in records]
        return {
            "unique_requests": len(records),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in usage),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in usage),
            "reported_cost_usd": sum(r.get("cost", 0) or 0 for r in usage),
            "cost_reported_requests": sum(r.get("cost") is not None for r in usage),
            "request_seconds": sum(r["wall_seconds"] for r in records),
            "response_models": sorted({r["response"].get("model", "") for r in records}),
        }
