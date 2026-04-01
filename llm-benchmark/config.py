# config.py

# ---------------------------
# Dataset
# ---------------------------
DATASET_PATH = "data/simple_qa_test_set.csv"


# ---------------------------
# Ollama Configuration
# ---------------------------
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"


# ---------------------------
# Models to Benchmark
# ---------------------------
MODEL_CONFIGS = [
    {
        "name": "llama3",
        "temperature": 0.2,
        "max_tokens": 512
    },
    {
        "name": "mistral",
        "temperature": 0.2,
        "max_tokens": 512
    },
    {
        "name": "phi3",
        "temperature": 0.2,
        "max_tokens": 512
    }
]


# ---------------------------
# Judge Model
# ---------------------------
JUDGE_MODEL = {
    "name": "llama3",
    "temperature": 0.0,
    "max_tokens": 512
}


# ---------------------------
# Evaluation Settings
# ---------------------------
USE_JUDGE = True
SAVE_RESULTS_PATH = "results/output.csv"


# ---------------------------
# Runtime Settings
# ---------------------------
REQUEST_TIMEOUT = 60
RETRY_COUNT = 2
VERBOSE = True