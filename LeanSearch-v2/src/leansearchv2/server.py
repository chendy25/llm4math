"""FastAPI server for LeanSearch v2 standard mode.

Run with::

    ./scripts/serve.sh                # serves the default mathlib corpus
    ./scripts/serve_physlib.sh        # serves both mathlib + physlib

or directly with uvicorn::

    python -m uvicorn leansearchv2.server:app --host 0.0.0.0 --port 8000

The server is multi-corpus: each request can select the corpus to search via
the `corpus` query parameter (e.g. `?corpus=physlib`). Corpus cuVS indexes
are loaded lazily on first request and kept in memory; embedding and
reranker models are loaded once at startup and shared across corpora.

Paths and tuning knobs are read from ``config.yaml`` at the repository root;
each can be overridden by an environment variable of the same name.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
from fastapi import Body, FastAPI, HTTPException, Query

from .config import (
    DEFAULT_CORPUS,
    corpus_cuvs_dir,
    corpus_config,
    get,
    get_path,
)
from .pipeline import (
    RetrievalPipeline,
    SearchResult,
    SharedModels,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-corpus index registry
# ---------------------------------------------------------------------------


class CorpusIndex:
    """Per-corpus state: vectordb dir + loaded cuVS index + texts/metadata.

    Embedding + reranker models live on the shared ``SharedModels`` instance
    and are reused across corpora (same Qwen3 weights, just different cuVS
    index + metadata)."""
    def __init__(self, name: str, vectordb_dir: str) -> None:
        self.name = name
        self.vectordb_dir = Path(vectordb_dir)
        self._loaded_pipeline: RetrievalPipeline | None = None
        self._lock = threading.Lock()

    def ensure_loaded(self, shared: SharedModels) -> RetrievalPipeline:
        if self._loaded_pipeline is not None:
            return self._loaded_pipeline
        with self._lock:
            if self._loaded_pipeline is not None:
                return self._loaded_pipeline
            log.info(f"[corpus={self.name}] Loading cuVS index from {self.vectordb_dir}")
            t0 = time.time()
            self._loaded_pipeline = RetrievalPipeline(
                vectordb_dir=str(self.vectordb_dir),
                shared_models=shared,
            )
            log.info(f"[corpus={self.name}] Ready in {time.time() - t0:.1f}s")
            return self._loaded_pipeline


class CorpusRegistry:
    """Map of corpus name -> CorpusIndex. Discovered from config.yaml."""
    def __init__(self, requested: list[str]) -> None:
        self._indexes: dict[str, CorpusIndex] = {}
        for name in requested:
            try:
                cfg = corpus_config(name)
            except KeyError:
                raise RuntimeError(
                    f"corpus {name!r} requested but not configured under paths.corpora"
                )
            if "cuvs_db" not in cfg:
                raise RuntimeError(
                    f"corpus {name!r} has no cuvs_db in paths.corpora.{name}"
                )
            self._indexes[name] = CorpusIndex(name, corpus_cuvs_dir(name))
        if DEFAULT_CORPUS not in self._indexes:
            # Always register the default corpus so `?corpus=` is optional
            # and the existing single-corpus deployments keep working.
            self._indexes[DEFAULT_CORPUS] = CorpusIndex(
                DEFAULT_CORPUS, corpus_cuvs_dir(DEFAULT_CORPUS)
            )

    def available(self) -> list[str]:
        return sorted(self._indexes.keys())

    def get(self, name: str) -> CorpusIndex:
        idx = self._indexes.get(name)
        if idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown corpus {name!r}. Available: {self.available()}",
            )
        return idx


def _requested_corpora() -> list[str]:
    """Resolve the list of corpora this server should serve at startup.

    Reads `serve.corpora` from config.yaml (a list). Falls back to the
    legacy single-corpus behavior (just `mathlib`) when unset."""
    raw = get("CORPORA", "serve", "corpora")
    if raw is None:
        return [DEFAULT_CORPUS]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s) for s in raw if str(s).strip()]
    return [DEFAULT_CORPUS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    log.info("Starting...")
    num_gpus = get("NUM_GPUS", "serve", "num_gpus")
    gpu_memory_utilization = float(
        get("GPU_MEMORY_UTILIZATION", "serve", "gpu_memory_utilization", default=0.9)
    )

    # Load embedding + reranker once and share across all corpora.
    embedding_model_path = get_path("EMBEDDING_MODEL_PATH", "models", "embedder")
    reranker_model_path = get_path("RERANKER_MODEL_PATH", "models", "reranker")
    app.shared = SharedModels(
        embedding_model_path=embedding_model_path,
        reranker_model_path=reranker_model_path,
        num_gpus=int(num_gpus) if num_gpus is not None else None,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Discover corpora from config and create lazy loaders for each.
    app.corpora = CorpusRegistry(_requested_corpora())
    for name in app.corpora.available():
        log.info(f"[corpus={name}] registered (cuvs_db={corpus_cuvs_dir(name)})")

    log.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/search")
def search(
    query: list[str] = Body(...),
    num_results: int = Body(default=10),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None),
    corpus: str = Query(default=DEFAULT_CORPUS, description="Corpus to search."),
) -> list[list[SearchResult]]:
    t0 = time.time()
    idx = app.corpora.get(corpus)
    pipeline = idx.ensure_loaded(app.shared)
    results = [
        pipeline.search(q, num_results, rerank=rerank, retrieve_k=retrieve_k)
        for q in query
    ]
    log.info(
        f"/search | corpus={corpus} | {time.time() - t0:.2f}s | n={len(query)} | rerank={rerank} | {query}"
    )
    return results


@app.post("/search_with_profile")
def search_with_profile(
    query: list[str] = Body(...),
    num_results: int = Body(default=10),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None),
    corpus: str = Query(default=DEFAULT_CORPUS),
) -> dict:
    t0 = time.time()
    idx = app.corpora.get(corpus)
    pipeline = idx.ensure_loaded(app.shared)
    all_results = []
    all_profiles = []
    for q in query:
        results, profile = pipeline.search(
            q, num_results, return_profile=True, rerank=rerank, retrieve_k=retrieve_k,
        )
        all_results.append(results)
        all_profiles.append(profile)

    total_time = time.time() - t0
    log.info(f"/search_with_profile | corpus={corpus} | {total_time:.2f}s | n={len(query)} | rerank={rerank} | {query}")

    return {
        "results": all_results,
        "profiles": all_profiles,
        "total_request_time": total_time,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_backend": "cuvs",
        "available_corpora": app.corpora.available(),
        "loaded_corpora": [
            name for name, idx in app.corpora._indexes.items()
            if idx._loaded_pipeline is not None
        ],
    }
