# config.py
import os

from dotenv import load_dotenv

load_dotenv()

HF_MODEL_PATH_1 = os.getenv("HF_MODEL_PATH_1", "/path/to/downloaded-models/google/gemma-2-2b-it")
HF_MODEL_PATH_2 = os.getenv("HF_MODEL_PATH_2", "/path/to/downloaded-models/Qwen/Qwen2.5-3B-Instruct")
HF_MODEL_PATH_3 = os.getenv("HF_MODEL_PATH_3", "/path/to/downloaded-models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

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
    # {
    #     "name": "llama3",
    #     "provider": "ollama",
    #     "temperature": 0.2,
    #     "max_tokens": 512
    # },
    # {
    #     "name": "mistral",
    #     "provider": "ollama",
    #     "temperature": 0.2,
    #     "max_tokens": 512
    # },
    # {
    #     "name": "phi3",
    #     "provider": "ollama",
    #     "temperature": 0.2,
    #     "max_tokens": 512
    # },
    {
        "name": "gpt-4o",
        "provider": "openai",
        "temperature": 0.2,
        "max_tokens": 512
    },
    {
        "name": "openai/gpt-oss-20b",
        "provider": "huggingface",
        "path": HF_MODEL_PATH_1,
        "temperature": 1,
        "max_tokens": 512
    },
    {
        "name": "Qwen/Qwen3-8B",
        "provider": "huggingface",
        "path": HF_MODEL_PATH_2,
        "temperature": 1,
        "max_tokens": 512
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "provider": "huggingface",
        "path": HF_MODEL_PATH_3,
        "temperature": 1,
        "max_tokens": 512
    },
]


# ---------------------------
# Judge Model
# ---------------------------
JUDGE_MODEL = {
    "name": "llama3",
    "provider": "ollama",
    "temperature": 0.0,
    "max_tokens": 512
}


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
