"""Simple FastAPI application exposing the SQL vector search functionality."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .indexer import SqlVectorIndex, load_index


class SearchRequest(BaseModel):
    """Payload for the ``/search`` endpoint."""

    query: str = Field(..., description="SQL text to use as the similarity query.")
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of similar SQL files to return.",
    )
    index: Optional[str] = Field(
        None,
        description="Optional path to the index file. If omitted a default must be provided when creating the app.",
    )


class SearchResult(BaseModel):
    """Response payload item returned by the ``/search`` endpoint."""

    path: str
    score: float


class SearchResponse(BaseModel):
    """Container for search results."""

    results: list[SearchResult]


def create_app(default_index: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    default_index:
        Optional path to an index that should be used when no ``index`` field is
        provided with the request payload. If supplied, the index is loaded once
        and cached in memory.
    """

    app = FastAPI(title="SQL Vectorizer Search API", version="1.0.0")

    cache: Dict[Path, SqlVectorIndex] = {}
    default_index_path = Path(default_index).expanduser().resolve() if default_index else None

    def _get_index(path: Path) -> SqlVectorIndex:
        if path in cache:
            return cache[path]
        try:
            index = load_index(path)
        except FileNotFoundError as exc:  # pragma: no cover - passthrough
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - generic error propagation
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        cache[path] = index
        return index

    @app.post("/search", response_model=SearchResponse)
    def search(payload: SearchRequest) -> SearchResponse:
        if payload.index:
            index_path = Path(payload.index).expanduser().resolve()
        elif default_index_path is not None:
            index_path = default_index_path
        else:
            raise HTTPException(
                status_code=400,
                detail="Index path must be provided either in the request payload or when creating the app.",
            )

        index = _get_index(index_path)

        try:
            results = index.search(payload.query, top_k=payload.top_k, include_scores=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response = SearchResponse(
            results=[
                SearchResult(path=str(path), score=score)
                for path, score in results
            ]
        )
        return response

    return app


app = create_app()
