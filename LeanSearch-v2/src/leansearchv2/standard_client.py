"""HTTP client for the standard-mode retriever endpoint.

Reasoning mode and the prove task consume the retriever exclusively through
this client; nothing here imports `pipeline.py` (which pulls in cuVS/torch),
so the client is CPU-only and can run anywhere with network access.

The client speaks the unified LeanSearch-v2 /search protocol and is
corpus-agnostic: it forwards `corpus` as a query-string parameter so the
server can route the request to the right cuVS index (mathlib / physlib /
...). When `corpus` is omitted, the server falls back to its default
corpus (mathlib), which matches every pre-existing single-corpus
deployment.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from .config import get


class ResultData(BaseModel):
    module_name: list[str]
    kind: str
    name: list[str]
    signature: str
    type: str
    value: str | None = None
    docstring: str | None = None
    informal_name: str | None = None
    informal_description: str | None = None


class SearchResult(BaseModel):
    result: ResultData
    distance: float


class StandardClient:
    def __init__(
        self,
        url: str | None = None,
        timeout: float | None = None,
        corpus: str | None = None,
    ) -> None:
        url = url or get("RETRIEVER_URL", "retriever", "url", default="http://localhost:8000")
        self.url = url.rstrip("/")
        self.timeout = float(
            timeout if timeout is not None else get("RETRIEVER_TIMEOUT_S", "retriever", "timeout_s", default=60)
        )
        self.corpus = corpus

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
        corpus: str | None = None,
    ) -> list[SearchResult]:
        results = await self.search_batch([query], top_k, rerank=rerank, retrieve_k=retrieve_k, corpus=corpus)
        return results[0]

    async def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
        corpus: str | None = None,
    ) -> list[list[SearchResult]]:
        body: dict[str, Any] = {"query": queries, "num_results": top_k, "rerank": rerank}
        if retrieve_k is not None:
            body["retrieve_k"] = retrieve_k
        params: dict[str, Any] = {}
        effective_corpus = corpus if corpus is not None else self.corpus
        if effective_corpus:
            params["corpus"] = effective_corpus
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.url}/search", json=body, params=params or None)
            r.raise_for_status()
            data = r.json()
        return [[SearchResult.model_validate(item) for item in batch] for batch in data]

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.url}/health")
            r.raise_for_status()
            return r.json()
