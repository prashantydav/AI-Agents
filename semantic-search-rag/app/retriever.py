from __future__ import annotations

import numpy as np
import chromadb
from pathlib import Path
from typing import List

from .chunking import split_fixed, split_recursive, split_semantic
from .embeddings import EmbeddingCache, compute_source_hash, embed_texts
from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MMR_LAMBDA,
    TOP_K,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)

CHUNKERS = {
    "fixed": split_fixed,
    "recursive": split_recursive,
    "semantic": split_semantic,
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: List[str],
    candidate_texts: List[str],
    candidate_metadatas: List[dict],
    top_k: int = TOP_K,
    lambda_mult: float = MMR_LAMBDA,
) -> List[dict]:
    if candidate_embeddings.shape[0] == 0:
        return []

    scores = [
        _cosine_similarity(query_embedding, candidate_embeddings[idx])
        for idx in range(len(candidate_embeddings))
    ]

    selected_indices = []
    candidate_indices = list(range(len(candidate_ids)))

    while len(selected_indices) < min(top_k, len(candidate_indices)):
        if not selected_indices:
            selected = max(candidate_indices, key=lambda i: scores[i])
            selected_indices.append(selected)
            candidate_indices.remove(selected)
            continue

        mmr_values = []
        for idx in candidate_indices:
            similarity_to_query = scores[idx]
            similarity_to_selected = max(
                _cosine_similarity(candidate_embeddings[idx], candidate_embeddings[selected_idx])
                for selected_idx in selected_indices
            )
            mmr_values.append(lambda_mult * similarity_to_query - (1 - lambda_mult) * similarity_to_selected)

        selected = candidate_indices[int(np.argmax(mmr_values))]
        selected_indices.append(selected)
        candidate_indices.remove(selected)

    return [
        {
            "id": candidate_ids[idx],
            "text": candidate_texts[idx],
            "metadata": candidate_metadatas[idx],
            "score": float(scores[idx]),
        }
        for idx in selected_indices
    ]


class SemanticSearchStore:
    def __init__(self, strategy: str, documents: List[dict], persist_root: Path = CHROMA_PERSIST_DIR):
        self.strategy = strategy
        self.documents = documents
        self.source_hash = compute_source_hash(self.documents)
        
        # Compute model key based on current provider and model
        if EMBEDDING_PROVIDER == "openai":
            self.model_key = f"openai_{EMBEDDING_MODEL}"
        elif EMBEDDING_PROVIDER == "sentence-transformers":
            self.model_key = f"sentence-transformers_{SENTENCE_TRANSFORMER_MODEL.replace('/', '_')}"
        else:
            self.model_key = EMBEDDING_PROVIDER
        
        self.persist_root = persist_root / self.model_key / strategy
        self.persist_root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_root))
        self.collection_name = f"semantic_search_{strategy}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.chunks = []
        self.embeddings = np.zeros((0, 1), dtype=np.float32)
        self.id_to_index = {}

    def _chunk_documents(self) -> List[dict]:
        chunker = CHUNKERS.get(self.strategy, split_fixed)
        chunks = []
        for document in self.documents:
            raw_text = document.get("text", "")
            if not raw_text.strip():
                continue

            split_texts = chunker(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk_index, chunk_text in enumerate(split_texts):
                chunk_id = f"{document.get('id')}::{chunk_index}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source_id": document.get("id"),
                            "title": document.get("title", ""),
                            "strategy": self.strategy,
                        },
                    }
                )

        return chunks

    def _build_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        if self.collection.count() > 0:
            return
        if not self.chunks or self.embeddings.size == 0:
            return

        self.collection.add(
            ids=[chunk["id"] for chunk in self.chunks],
            documents=[chunk["text"] for chunk in self.chunks],
            metadatas=[chunk["metadata"] for chunk in self.chunks],
            embeddings=self.embeddings.tolist(),
        )

    def prepare(self, force: bool = False) -> None:
        cache = EmbeddingCache(self.strategy, self.source_hash, model_key=self.model_key)
        loaded = None if force else cache.load()

        if loaded is not None:
            self.chunks, self.embeddings = loaded
        else:
            self.chunks = self._chunk_documents()
            texts = [chunk["text"] for chunk in self.chunks]
            self.embeddings = embed_texts(texts, provider=EMBEDDING_PROVIDER)
            cache.save(self.chunks, self.embeddings)

        self.id_to_index = {chunk["id"]: idx for idx, chunk in enumerate(self.chunks)}
        self._build_collection()

    def query(self, query_text: str, top_k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA, candidates: int = 20) -> List[dict]:
        if self.collection.count() == 0:
            self.prepare()

        query_embedding = embed_texts([query_text], provider=EMBEDDING_PROVIDER)[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidates,
            include=["metadatas", "documents"],
        )

        candidate_ids = results["ids"][0]
        candidate_texts = results["documents"][0]
        candidate_metadatas = results["metadatas"][0]

        candidate_embeddings = np.vstack([
            self.embeddings[self.id_to_index[item_id]] for item_id in candidate_ids
        ])

        return _mmr_rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            candidate_metadatas=candidate_metadatas,
            top_k=top_k,
            lambda_mult=mmr_lambda,
        )


class SemanticSearchEngine:
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.stores: dict[str, SemanticSearchStore] = {}

    def get_store(self, strategy: str) -> SemanticSearchStore:
        if strategy not in self.stores:
            self.stores[strategy] = SemanticSearchStore(strategy, self.documents)
        return self.stores[strategy]

    def query(self, query_text: str, strategy: str = "semantic", top_k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA) -> List[dict]:
        store = self.get_store(strategy)
        store.prepare()
        return store.query(query_text, top_k=top_k, mmr_lambda=mmr_lambda)

    def reload(self) -> None:
        self.stores = {}
