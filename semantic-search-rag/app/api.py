from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from .cache import RedisCache
from .embeddings import current_model_key, embed_texts, get_embedding_readiness
from .retriever import SemanticSearchEngine
from .rag import answer_query
from .utils import load_documents, to_serializable
from config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    REDIS_CHAT_TTL,
    REDIS_QUERY_EMBEDDING_TTL,
    REDIS_SEARCH_TTL,
    SENTENCE_TRANSFORMER_MODEL,
)

ALLOWED_STRATEGIES = ["fixed", "recursive", "semantic"]

app = FastAPI(
    title="Semantic Search RAG API",
    description="FastAPI backend for semantic search, retrieval reranking, and RAG-powered response generation.",
)


class QueryPayload(BaseModel):
    query: str = Field(..., min_length=5)
    strategy: str = Field("semantic")
    top_k: int = Field(5, ge=1, le=10)


class ChatPayload(QueryPayload):
    pass


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: List[SearchResult]


class ChatResponse(BaseModel):
    query: str
    strategy: str
    answer: str
    sources: List[SearchResult]


def _current_embedding_config() -> dict:
    if EMBEDDING_PROVIDER == "openai":
        return {"provider": "openai", "model": EMBEDDING_MODEL}
    if EMBEDDING_PROVIDER == "sentence-transformers":
        return {"provider": "sentence-transformers", "model": SENTENCE_TRANSFORMER_MODEL}
    return {"provider": EMBEDDING_PROVIDER}


def _serialize_embedding(vector) -> list[float]:
    return [float(value) for value in vector]


def _get_query_embedding(cache: RedisCache, query: str) -> list[float]:
    model_key = current_model_key()
    cache_key = cache.build_key("qembed", EMBEDDING_PROVIDER, model_key, query)
    cached = cache.get_json(cache_key)
    if cached is not None:
        return cached

    embedding = _serialize_embedding(embed_texts([query], provider=EMBEDDING_PROVIDER)[0])
    cache.set_json(cache_key, embedding, REDIS_QUERY_EMBEDDING_TTL)
    return embedding


@app.on_event("startup")
async def startup_event():
    documents = load_documents()
    app.state.search_engine = SemanticSearchEngine(documents)
    app.state.embedding_readiness = get_embedding_readiness()
    app.state.cache = RedisCache()


@app.get("/health")
async def health():
    readiness = app.state.embedding_readiness
    return {
        "status": "ok" if readiness.get("ready") else "degraded",
        "embedding": readiness,
    }


@app.get("/config")
async def get_config():
    return {
        "embedding": {
            **_current_embedding_config(),
            "readiness": app.state.embedding_readiness,
        },
        "chunking_strategies": ALLOWED_STRATEGIES,
        "cache": {
            "redis_enabled": app.state.cache.enabled,
            "corpus_hash": app.state.search_engine.corpus_hash,
        },
    }


@app.post("/search", response_model=SearchResponse)
async def search(payload: QueryPayload):
    strategy = payload.strategy.lower().strip()
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Supported strategies: {ALLOWED_STRATEGIES}")

    engine: SemanticSearchEngine = app.state.search_engine
    cache: RedisCache = app.state.cache
    search_cache_key = cache.build_key(
        "search",
        engine.corpus_hash,
        strategy,
        payload.top_k,
        EMBEDDING_PROVIDER,
        current_model_key(),
        payload.query,
    )
    cached = cache.get_json(search_cache_key)
    if cached is not None:
        return cached

    try:
        query_embedding = _get_query_embedding(cache, payload.query)
        results = engine.query_by_embedding(query_embedding, strategy=strategy, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = {
        "query": payload.query,
        "strategy": strategy,
        "results": [to_serializable(result) for result in results],
    }
    cache.set_json(search_cache_key, response, REDIS_SEARCH_TTL)
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatPayload):
    strategy = payload.strategy.lower().strip()
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Supported strategies: {ALLOWED_STRATEGIES}")

    engine: SemanticSearchEngine = app.state.search_engine
    cache: RedisCache = app.state.cache
    chat_cache_key = cache.build_key(
        "chat",
        engine.corpus_hash,
        strategy,
        payload.top_k,
        EMBEDDING_PROVIDER,
        current_model_key(),
        payload.query,
    )
    cached = cache.get_json(chat_cache_key)
    if cached is not None:
        return cached

    try:
        query_embedding = _get_query_embedding(cache, payload.query)
        results = engine.query_by_embedding(query_embedding, strategy=strategy, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer = answer_query(payload.query, results)
    response = {
        "query": payload.query,
        "strategy": strategy,
        "answer": answer,
        "sources": [to_serializable(result) for result in results],
    }
    cache.set_json(chat_cache_key, response, REDIS_CHAT_TTL)
    return response


@app.post("/reload")
async def reload_store():
    documents = load_documents()
    engine = SemanticSearchEngine(documents)
    app.state.search_engine = engine
    app.state.embedding_readiness = get_embedding_readiness()
    return {"status": "reloaded", "embedding": app.state.embedding_readiness}
