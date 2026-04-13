# Semantic Search Engine with Full RAG Pipeline

This project implements a production-ready semantic search system with:
- 3 chunking strategies: `fixed`, `recursive`, `semantic`
- Dual embedding support: OpenAI embeddings or Sentence Transformers (E5-Large-v2 multilingual)
- Batch processing and intelligent caching for embeddings
- ChromaDB vector store persistence with model-scoped namespaces
- MMR reranking for retrieval diversity and relevance
- FastAPI backend for search/chat endpoints
- Streamlit UI showing retrieved source context with answers

## Setup

1. Install dependencies:

```bash
cd /home/prashant/AI-Agents/semantic-search-rag
python -m pip install -r requirements.txt
```

2. Configure the app with the root `.env` file.

The repository now includes a `.env` file containing all configurable constants from `config.py`.
Update the values there before running the app.

**Default `.env` values:**
```dotenv
OPENAI_API_KEY=
BACKEND_URL=http://localhost:8000
REDIS_URL=redis://localhost:6379/0
CHROMA_PERSIST_DIR=/home/prashant/AI-Agents/semantic-search-rag/chroma_db
EMBEDDING_CACHE_DIR=/home/prashant/AI-Agents/semantic-search-rag/cache
DATA_DIR=/home/prashant/AI-Agents/semantic-search-rag/data
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
SENTENCE_TRANSFORMER_MODEL=intfloat/e5-large-v2
SENTENCE_TRANSFORMER_DEVICE=cpu
OPENAI_COMPLETION_MODEL=gpt-3.5-turbo
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K=5
MMR_LAMBDA=0.6
BATCH_SIZE=16
MAX_CONCURRENCY_WORKERS=4
REDIS_QUERY_EMBEDDING_TTL=3600
REDIS_SEARCH_TTL=300
REDIS_CHAT_TTL=300
```

3. Choose your embedding model: OpenAI (default) or Sentence Transformers

**For OpenAI:**
```bash
OPENAI_API_KEY="your_openai_api_key"
EMBEDDING_PROVIDER="openai"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
```

**For Sentence Transformers (E5-Large multilingual, no API key needed):**
```bash
EMBEDDING_PROVIDER="sentence-transformers"
SENTENCE_TRANSFORMER_MODEL="intfloat/e5-large-v2"
SENTENCE_TRANSFORMER_DEVICE="cuda"  # or "cpu"
```

4. Put your source files inside `data/` or point ingestion at a specific file/folder.

Supported formats:
- `csv`
- `xlsx`
- `xls`
- `pdf`
- `docx`
- `txt`
- `md`
- `json`

The loader will scan directories recursively and normalize those sources into the internal document format used by the RAG pipeline.

5. Ingest the corpus:

```bash
python ingest.py --strategy semantic
```

Examples:

```bash
python ingest.py --source data
python ingest.py --source data/company-handbook.pdf
python ingest.py --source data/quarterly_reports
python ingest.py --source data/sales.xlsx
```

## Run the backend

```bash
cd /home/prashant/AI-Agents/semantic-search-rag
uvicorn app.api:app --reload --port 8000
```

The API will automatically use the configured embedding provider.

## Run the Streamlit UI

```bash
cd /home/prashant/AI-Agents/semantic-search-rag
streamlit run streamlit_app.py
```

## API endpoints

- `GET /health` - health check
- `GET /config` - show current embedding model configuration
- `POST /search` - retrieve top chunks for a query
- `POST /chat` - generate a RAG-powered answer with cited sources
- `POST /reload` - drop cached stores and reload

## Performance

The backend now uses:
- concurrent preparation of per-source vector stores
- concurrent fan-out querying across source stores
- Redis caching for query embeddings, search responses, and chat responses
- cache keys scoped by corpus hash, embedding provider, and model so changed source content invalidates old query caches

If Redis is not installed or not reachable, the application falls back to filesystem embedding cache plus Chroma persistence.

## Source handling

How each format is converted:
- `csv`: if columns `id`, `title`, and `text` exist, each row becomes a document; otherwise each row is flattened into key/value text
- `xlsx` and `xls`: each row in each sheet becomes a document
- `pdf`: each page becomes a document
- `docx`: the full document text becomes a document
- `txt` and `md`: the full file becomes a document
- `json`: tabular JSON arrays are flattened into document text

Each retrieved chunk keeps source metadata such as file path, file type, and when applicable page number, sheet name, or row number.

## Model comparison

| Model | Type | Latency | Cost | Multilingual | Local |
|-------|------|---------|------|--------------|-------|
| text-embedding-3-large | OpenAI | Fast | $$ | Yes | No |
| E5-Large-v2 | Sentence Transformers | Slower | Free | Yes | Yes |

## Embedding Cache & Storage

Each embedding model gets its own namespace:
- `cache/{model_key}/{source_hash}/{strategy}/` - embedding cache with metadata
- `chroma_db/{model_key}/{source_hash}/{strategy}/` - ChromaDB persistent storage

This allows you to switch between embedding models without collision.

For source-level storage, the runtime now creates one vector database namespace per unique source content hash. If the same source is ingested again without content changes, the existing cache/vector store is reused and a duplicate database is not created.

## Notes

- First ingestion may take time: OpenAI depends on API, Sentence Transformers depends on model download & inference speed
- Cache and persistent data are automatically organized by model type
- You can switch chunking strategies and embedding models independently
- Device selection (CPU/CUDA) can be set via `SENTENCE_TRANSFORMER_DEVICE`
