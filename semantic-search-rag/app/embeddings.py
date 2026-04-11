import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import openai

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    SENTENCE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_DEVICE,
)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _safe_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"
    except Exception as exc:
        return f"unavailable ({exc})"


def _sentence_transformers_diagnostics() -> str:
    package_versions = {
        "sentence-transformers": _safe_package_version("sentence-transformers"),
        "transformers": _safe_package_version("transformers"),
        "huggingface_hub": _safe_package_version("huggingface_hub"),
        "packaging": _safe_package_version("packaging"),
    }
    return ", ".join(f"{name}={package_versions[name]}" for name in package_versions)


def _format_sentence_transformer_init_error(exc: Exception, model: str) -> str:
    message = str(exc)
    network_indicators = (
        "Temporary failure in name resolution",
        "Name or service not known",
        "Failed to establish a new connection",
        "Connection error",
        "client has been closed",
    )
    if any(indicator.lower() in message.lower() for indicator in network_indicators):
        return (
            "Failed to initialize the sentence-transformers model. "
            "The configured model is not available in the local cache and the runtime could not reach Hugging Face to download it. "
            f"Download `{model}` in an environment with internet access, or switch to the `openai` embedding provider."
        )

    return (
        "Failed to initialize the sentence-transformers model. "
        "Ensure sentence-transformers, transformers, and their dependencies are installed and compatible with your Python environment. "
        "If you are using a newer Python version such as 3.12, consider creating a clean virtual environment and reinstalling the packages. "
    )


def get_embedding_readiness(
    provider: str = EMBEDDING_PROVIDER,
    model: Optional[str] = None,
    device: str = SENTENCE_TRANSFORMER_DEVICE,
) -> Dict[str, Any]:
    if provider == "openai":
        selected_model = model or EMBEDDING_MODEL
        ready = bool(OPENAI_API_KEY)
        return {
            "provider": provider,
            "model": selected_model,
            "ready": ready,
            "detail": (
                "OpenAI API key detected."
                if ready
                else "OPENAI_API_KEY is not configured."
            ),
        }

    if provider == "sentence-transformers":
        selected_model = model or SENTENCE_TRANSFORMER_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            return {
                "provider": provider,
                "model": selected_model,
                "device": device,
                "ready": False,
                "detail": (
                    "sentence-transformers is unavailable in this environment. "
                    f"Detected package versions: {_sentence_transformers_diagnostics()}. "
                    f"Original error: {exc}"
                ),
            }

        try:
            SentenceTransformer(selected_model, device=device, local_files_only=True)
        except Exception as exc:
            return {
                "provider": provider,
                "model": selected_model,
                "device": device,
                "ready": False,
                "detail": (
                    _format_sentence_transformer_init_error(exc, selected_model)
                    + f" Detected package versions: {_sentence_transformers_diagnostics()}."
                ),
            }

        return {
            "provider": provider,
            "model": selected_model,
            "device": device,
            "ready": True,
            "detail": "Sentence-transformers model is available in the local cache.",
        }

    return {
        "provider": provider,
        "model": model,
        "ready": False,
        "detail": f"Unknown embedding provider: {provider}. Supported: openai, sentence-transformers",
    }


def compute_source_hash(documents: List[dict]) -> str:
    combined = "||".join(
        f"{doc.get('id','')}::{doc.get('title','')}::{doc.get('text','')}" for doc in documents
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(
        self,
        strategy: str,
        source_hash: str,
        model_key: str = None,
        cache_root: Path = CACHE_DIR,
    ):
        self.strategy = strategy
        self.source_hash = source_hash
        self.model_key = model_key or f"{EMBEDDING_PROVIDER}_{EMBEDDING_MODEL}"
        # Create cache directory scoped by strategy and model
        self.cache_dir = cache_root / self.model_key / strategy
        _ensure_dir(self.cache_dir)
        self.meta_path = self.cache_dir / "meta.json"
        self.chunks_path = self.cache_dir / "chunks.json"
        self.embeddings_path = self.cache_dir / "embeddings.npy"

    def is_valid(self) -> bool:
        if not self.meta_path.exists() or not self.chunks_path.exists() or not self.embeddings_path.exists():
            return False

        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return meta.get("source_hash") == self.source_hash and meta.get("model_key") == self.model_key
        except Exception:
            return False

    def load(self) -> Optional[Tuple[List[dict], np.ndarray]]:
        if not self.is_valid():
            return None

        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(self.embeddings_path)
        return chunks, embeddings

    def save(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        self.chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        np.save(self.embeddings_path, embeddings)
        self.meta_path.write_text(
            json.dumps(
                {
                    "strategy": self.strategy,
                    "source_hash": self.source_hash,
                    "model_key": self.model_key,
                    "count": len(chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def embed_texts_openai(texts: List[str], batch_size: int = BATCH_SIZE, model: str = EMBEDDING_MODEL) -> np.ndarray:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to compute embeddings.")

    openai.api_key = OPENAI_API_KEY
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = openai.Embedding.create(model=model, input=batch)
        embeddings.extend([item["embedding"] for item in response["data"]])

    return np.asarray(embeddings, dtype=np.float32)


def embed_texts_sentence_transformers(
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    model: str = SENTENCE_TRANSFORMER_MODEL,
    device: str = SENTENCE_TRANSFORMER_DEVICE,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Unable to import sentence-transformers. "
            "This can happen when sentence-transformers or its transformers dependencies are not installed or are incompatible with the Python environment. "
            "Install or upgrade them with: pip install --upgrade sentence-transformers transformers packaging huggingface_hub"
            f"\nDetected package versions: {_sentence_transformers_diagnostics()}"
            f"\nOriginal error: {exc}"
        ) from exc

    try:
        transformer = SentenceTransformer(model, device=device)
    except Exception as exc:
        raise RuntimeError(
            _format_sentence_transformer_init_error(exc, model)
            + f"\nDetected package versions: {_sentence_transformers_diagnostics()}"
            + f"\nOriginal error: {exc}"
        ) from exc

    embeddings = transformer.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_texts(
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    provider: str = EMBEDDING_PROVIDER,
    model: str = None,
) -> np.ndarray:
    if provider == "openai":
        model = model or EMBEDDING_MODEL
        return embed_texts_openai(texts, batch_size=batch_size, model=model)
    elif provider == "sentence-transformers":
        model = model or SENTENCE_TRANSFORMER_MODEL
        return embed_texts_sentence_transformers(texts, batch_size=batch_size, model=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Supported: openai, sentence-transformers")
