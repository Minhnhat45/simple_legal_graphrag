from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from .config import Settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.chat_model = settings.ollama_chat_model
        self.embed_model = settings.ollama_embed_model
        self.timeout = settings.ollama_timeout_seconds

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._request(
            "/api/embed",
            {"model": self.embed_model, "input": list(texts), "truncate": True},
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an invalid embedding response")
        return [[float(value) for value in vector] for vector in embeddings]

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        response = self._request(
            "/api/chat",
            {
                "model": self.chat_model,
                "messages": list(messages),
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty chat response")
        return content.strip()

    def health(self) -> bool:
        response = self._request("/api/tags")
        return isinstance(response.get("models"), list)
