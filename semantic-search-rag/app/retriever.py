from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import chromadb
from pathlib import Path
from typing import Dict, List

from .chunking import split_fixed, split_recursive, split_semantic
from .embeddings import EmbeddingCache, compute_source_hash, current_model_key, embed_texts
from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CONCURRENCY_WORKERS,
    MMR_LAMBDA,
    TOP_K,
    EMBEDDING_PROVIDER,
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
        self.source_path = str(self.documents[0].get("source_path", self.documents[0].get("id", "source")))
        self.source_name = Path(self.source_path).stem or self.source_hash[:12]
        self.source_key = self.source_hash
        self.model_key = current_model_key()
        
        self.persist_root = persist_root / self.model_key / self.source_key / strategy
        self.persist_root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_root))
        self.collection_name = f"semantic_search_{strategy}_{self.source_hash[:12]}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.manifest_path = self.persist_root / "source.json"
        self.chunks = []
        self.embeddings = np.zeros((0, 1), dtype=np.float32)
        self.id_to_index = {}

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "source_hash": self.source_hash,
                    "source_path": self.source_path,
                    "source_name": self.source_name,
                    "strategy": self.strategy,
                    "model_key": self.model_key,
                    "document_count": len(self.documents),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _chunk_documents(self) -> List[dict]:
        chunker = CHUNKERS.get(self.strategy, split_fixed)
        chunks = []
        for document in self.documents:
            raw_text = document.get("text", "")
            if not raw_text.strip():
                continue

            split_texts = chunker(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk_index, chunk_text in enumerate(split_texts):
                chunk_id = f"{self.source_hash}::{document.get('id')}::{chunk_index}"
                metadata = {
                    "source_hash": self.source_hash,
                    "source_id": document.get("id"),
                    "title": document.get("title", ""),
                    "strategy": self.strategy,
                }
                for field in ("source_path", "source_type", "page", "sheet", "row"):
                    if document.get(field) is not None:
                        metadata[field] = document.get(field)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata,
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
        cache = EmbeddingCache(
            self.strategy,
            self.source_hash,
            namespace=self.source_key,
            model_key=self.model_key,
        )
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
        self._write_manifest()

    def query_candidates(self, query_embedding: np.ndarray, candidates: int = 20) -> Dict[str, List]:
        if self.collection.count() == 0:
            self.prepare()

        if self.collection.count() == 0:
            return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidates, self.collection.count()),
            include=["metadatas", "documents"],
        )

        candidate_ids = results["ids"][0]
        candidate_texts = results["documents"][0]
        candidate_metadatas = results["metadatas"][0]
        if not candidate_ids:
            return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}

        candidate_embeddings = [
            self.embeddings[self.id_to_index[item_id]] for item_id in candidate_ids
        ]

        return {
            "ids": candidate_ids,
            "texts": candidate_texts,
            "metadatas": candidate_metadatas,
            "embeddings": candidate_embeddings,
        }

    def query(self, query_text: str, top_k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA, candidates: int = 20) -> List[dict]:
        query_embedding = embed_texts([query_text], provider=EMBEDDING_PROVIDER)[0]
        candidates_payload = self.query_candidates(query_embedding, candidates=candidates)
        if not candidates_payload["ids"]:
            return []

        return _mmr_rerank(
            query_embedding=query_embedding,
            candidate_embeddings=np.vstack(candidates_payload["embeddings"]),
            candidate_ids=candidates_payload["ids"],
            candidate_texts=candidates_payload["texts"],
            candidate_metadatas=candidates_payload["metadatas"],
            top_k=top_k,
            lambda_mult=mmr_lambda,
        )


class SemanticSearchEngine:
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.documents_by_source = self._group_documents_by_source(documents)
        self.stores: dict[str, SemanticSearchStore] = {}
        self.corpus_hash = self._compute_corpus_hash()

    @staticmethod
    def _group_documents_by_source(documents: List[dict]) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {}
        for document in documents:
            source_path = document.get("source_path") or document.get("id")
            grouped.setdefault(str(source_path), []).append(document)

        deduplicated: Dict[str, List[dict]] = {}
        seen_hashes = set()
        for source_path, source_documents in grouped.items():
            source_hash = compute_source_hash(source_documents)
            if source_hash in seen_hashes:
                continue
            deduplicated[source_path] = source_documents
            seen_hashes.add(source_hash)

        return deduplicated

    def get_stores(self, strategy: str) -> List[SemanticSearchStore]:
        stores = []
        for source_documents in self.documents_by_source.values():
            source_hash = compute_source_hash(source_documents)
            store_key = f"{strategy}:{source_hash}"
            if store_key not in self.stores:
                self.stores[store_key] = SemanticSearchStore(strategy, source_documents)
            stores.append(self.stores[store_key])
        return stores

    def _compute_corpus_hash(self) -> str:
        source_hashes = sorted(
            compute_source_hash(source_documents)
            for source_documents in self.documents_by_source.values()
        )
        return compute_source_hash(
            [{"id": str(index), "title": "", "text": source_hash} for index, source_hash in enumerate(source_hashes)]
        )

    @staticmethod
    def _run_parallel(items, fn):
        if not items:
            return []
        worker_count = min(MAX_CONCURRENCY_WORKERS, len(items))
        if worker_count <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(fn, items))

    def get_store(self, strategy: str) -> SemanticSearchStore:
        stores = self.get_stores(strategy)
        if not stores:
            raise ValueError("No sources available to build a semantic search store.")
        return stores[0]

    def prepare(self, strategy: str = "semantic", force: bool = False) -> List[SemanticSearchStore]:
        stores = self.get_stores(strategy)
        self._run_parallel(stores, lambda store: store.prepare(force=force))
        return stores

    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        strategy: str = "semantic",
        top_k: int = TOP_K,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> List[dict]:
        candidate_ids = []
        candidate_texts = []
        candidate_metadatas = []
        candidate_embeddings = []

        stores = self.get_stores(strategy)
        self._run_parallel(stores, lambda store: store.prepare())
        payloads = self._run_parallel(stores, lambda store: store.query_candidates(query_embedding))

        for payload in payloads:
            candidate_ids.extend(payload["ids"])
            candidate_texts.extend(payload["texts"])
            candidate_metadatas.extend(payload["metadatas"])
            candidate_embeddings.extend(payload["embeddings"])

        if not candidate_ids:
            return []

        return _mmr_rerank(
            query_embedding=query_embedding,
            candidate_embeddings=np.vstack(candidate_embeddings),
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            candidate_metadatas=candidate_metadatas,
            top_k=top_k,
            lambda_mult=mmr_lambda,
        )

    def query(self, query_text: str, strategy: str = "semantic", top_k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA) -> List[dict]:
        query_embedding = embed_texts([query_text], provider=EMBEDDING_PROVIDER)[0]
        return self.query_by_embedding(query_embedding, strategy=strategy, top_k=top_k, mmr_lambda=mmr_lambda)

    def reload(self) -> None:
        self.stores = {}
