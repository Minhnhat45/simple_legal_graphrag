from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import QueryFilters
from .service import GraphRAGService


class FilterRequest(BaseModel):
    document_type: str = ""
    sector: str = ""
    field: str = ""
    status: str = ""
    agency: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    filters: FilterRequest | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


@lru_cache(maxsize=1)
def get_service() -> GraphRAGService:
    return GraphRAGService.from_settings()


app = FastAPI(
    title="Vietnamese Legal GraphRAG",
    version="0.1.0",
    description="Hybrid Qdrant + FalkorDB retrieval over Vietnamese legal documents.",
)


@app.get("/health")
def health() -> dict[str, object]:
    components = get_service().health()
    return {"ok": all(components.values()), "components": components}


@app.get("/v1/stats")
def stats() -> dict[str, object]:
    try:
        return get_service().stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, object]:
    filters = QueryFilters(**request.filters.model_dump()) if request.filters else QueryFilters()
    try:
        return (
            get_service()
            .answer(
                request.question,
                filters=filters,
                top_k=request.top_k,
            )
            .to_dict()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
