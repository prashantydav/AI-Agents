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

4. Ingest the sample corpus:

```bash
python ingest.py --strategy semantic
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

## Corpus format

The sample data file is at `data/sample_docs.csv` and expects columns:
- `id`
- `title`
- `text`

## Model comparison

| Model | Type | Latency | Cost | Multilingual | Local |
|-------|------|---------|------|--------------|-------|
| text-embedding-3-large | OpenAI | Fast | $$ | Yes | No |
| E5-Large-v2 | Sentence Transformers | Slower | Free | Yes | Yes |

## Embedding Cache & Storage

Each embedding model gets its own namespace:
- `cache/{model_key}/{strategy}/` - embedding cache with metadata
- `chroma_db/{model_key}/{strategy}/` - ChromaDB persistent storage

This allows you to switch between embedding models without collision.

## Notes

- First ingestion may take time: OpenAI depends on API, Sentence Transformers depends on model download & inference speed
- Cache and persistent data are automatically organized by model type
- You can switch chunking strategies and embedding models independently
- Device selection (CPU/CUDA) can be set via `SENTENCE_TRANSFORMER_DEVICE`
