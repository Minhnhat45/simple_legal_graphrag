from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LegalDocument:
    id: str
    title: str
    full_text: str
    number: str
    document_type: str
    sector: str
    issued_date: str
    field: str
    effective_date: str
    status: str
    expiry_date: str
    agency: str
    signer_position: str
    signer: str

    @property
    def fingerprint(self) -> str:
        values = (
            self.id,
            self.title,
            self.full_text,
            self.number,
            self.document_type,
            self.sector,
            self.issued_date,
            self.field,
            self.effective_date,
            self.status,
            self.expiry_date,
            self.agency,
            self.signer_position,
            self.signer,
        )
        return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    document_id: str
    index: int
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class Mention:
    document_id: str
    number: str
    number_norm: str
    kinds: tuple[str, ...]
    count: int


@dataclass(frozen=True, slots=True)
class QueryFilters:
    document_type: str = ""
    sector: str = ""
    field: str = ""
    status: str = ""
    agency: str = ""

    def as_payload_map(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "document_type": self.document_type,
                "sector": self.sector,
                "field": self.field,
                "status": self.status,
                "agency": self.agency,
            }.items()
            if value
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    title: str
    number: str
    section: str
    text: str
    vector_score: float
    final_score: float = 0.0


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    document_id: str
    title: str
    number: str
    graph_score: float
    relations: tuple[str, ...]
    seed_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    hits: tuple[SearchHit, ...]
    graph_neighbors: tuple[GraphNeighbor, ...]


@dataclass(slots=True)
class Source:
    citation: str
    chunk_id: str
    document_id: str
    title: str
    number: str
    section: str
    text: str
    score: float
    cited: bool = False


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    sources: tuple[Source, ...]
    graph_context: tuple[GraphNeighbor, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [asdict(source) for source in self.sources],
            "graph_context": [asdict(item) for item in self.graph_context],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class IngestReport:
    source: str
    documents_seen: int
    documents_indexed: int
    documents_skipped: int
    chunks_indexed: int
    mentions_indexed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
