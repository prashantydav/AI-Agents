import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT_DIR / ".env")


def _get_int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma_db")))
CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", str(ROOT_DIR / "cache")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))

# Embedding model selection: "openai" or "sentence-transformers"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# OpenAI embeddings
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Sentence Transformers embeddings
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "intfloat/e5-large-v2")
SENTENCE_TRANSFORMER_DEVICE = os.getenv("SENTENCE_TRANSFORMER_DEVICE", "cpu")

COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", 200)
TOP_K = _get_int_env("TOP_K", 5)
MMR_LAMBDA = _get_float_env("MMR_LAMBDA", 0.6)
BATCH_SIZE = _get_int_env("BATCH_SIZE", 16)
