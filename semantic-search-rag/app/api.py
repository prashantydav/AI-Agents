from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from .retriever import SemanticSearchEngine
from .embeddings import get_embedding_readiness
from .rag import answer_query
from .utils import load_documents, to_serializable
from config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, SENTENCE_TRANSFORMER_MODEL

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


@app.on_event("startup")
async def startup_event():
    documents = load_documents()
    app.state.search_engine = SemanticSearchEngine(documents)
    app.state.embedding_readiness = get_embedding_readiness()


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
    }


@app.post("/search", response_model=SearchResponse)
async def search(payload: QueryPayload):
    strategy = payload.strategy.lower().strip()
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Supported strategies: {ALLOWED_STRATEGIES}")

    engine: SemanticSearchEngine = app.state.search_engine
    try:
        results = engine.query(payload.query, strategy=strategy, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "query": payload.query,
        "strategy": strategy,
        "results": [to_serializable(result) for result in results],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatPayload):
    strategy = payload.strategy.lower().strip()
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Supported strategies: {ALLOWED_STRATEGIES}")

    engine: SemanticSearchEngine = app.state.search_engine
    try:
        results = engine.query(payload.query, strategy=strategy, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer = answer_query(payload.query, results)
    return {
        "query": payload.query,
        "strategy": strategy,
        "answer": answer,
        "sources": [to_serializable(result) for result in results],
    }


@app.post("/reload")
async def reload_store():
    engine: SemanticSearchEngine = app.state.search_engine
    engine.reload()
    app.state.embedding_readiness = get_embedding_readiness()
    return {"status": "reloaded", "embedding": app.state.embedding_readiness}
