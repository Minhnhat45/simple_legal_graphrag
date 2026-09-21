from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load a small, dependency-free subset of dotenv syntax."""

    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _text(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _integer(name: str, default: int) -> int:
    raw = _text(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "legal-graphrag"
    neo4j_database: str = "neo4j"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "vn_legal_chunks"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e4b"
    ollama_embed_model: str = "bge-m3"
    ollama_timeout_seconds: int = 180

    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250
    embed_batch_size: int = 16
    document_batch_size: int = 16
    retrieval_top_k: int = 10
    graph_expansion_limit: int = 12
    max_context_chars: int = 18_000
    max_chunks_per_document: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        settings = cls(
            neo4j_uri=_text("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=_text("NEO4J_USERNAME", "neo4j"),
            neo4j_password=_text("NEO4J_PASSWORD", "legal-graphrag"),
            neo4j_database=_text("NEO4J_DATABASE", "neo4j"),
            qdrant_url=_text("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_text("QDRANT_API_KEY", ""),
            qdrant_collection=_text("QDRANT_COLLECTION", "vn_legal_chunks"),
            ollama_base_url=_text("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_chat_model=_text("OLLAMA_CHAT_MODEL", "gemma4:e4b"),
            ollama_embed_model=_text("OLLAMA_EMBED_MODEL", "bge-m3"),
            ollama_timeout_seconds=_integer("OLLAMA_TIMEOUT_SECONDS", 180),
            chunk_size_chars=_integer("CHUNK_SIZE_CHARS", 1800),
            chunk_overlap_chars=_integer("CHUNK_OVERLAP_CHARS", 250),
            embed_batch_size=_integer("EMBED_BATCH_SIZE", 16),
            document_batch_size=_integer("DOCUMENT_BATCH_SIZE", 16),
            retrieval_top_k=_integer("RETRIEVAL_TOP_K", 10),
            graph_expansion_limit=_integer("GRAPH_EXPANSION_LIMIT", 12),
            max_context_chars=_integer("MAX_CONTEXT_CHARS", 18_000),
            max_chunks_per_document=_integer("MAX_CHUNKS_PER_DOCUMENT", 2),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        positive = {
            "OLLAMA_TIMEOUT_SECONDS": self.ollama_timeout_seconds,
            "CHUNK_SIZE_CHARS": self.chunk_size_chars,
            "EMBED_BATCH_SIZE": self.embed_batch_size,
            "DOCUMENT_BATCH_SIZE": self.document_batch_size,
            "RETRIEVAL_TOP_K": self.retrieval_top_k,
            "GRAPH_EXPANSION_LIMIT": self.graph_expansion_limit,
            "MAX_CONTEXT_CHARS": self.max_context_chars,
            "MAX_CHUNKS_PER_DOCUMENT": self.max_chunks_per_document,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.chunk_overlap_chars < 0:
            raise ValueError("CHUNK_OVERLAP_CHARS cannot be negative")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
