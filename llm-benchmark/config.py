# config.py
import os

from dotenv import load_dotenv

load_dotenv()

HF_MODEL_PATH_1 = os.getenv("HF_MODEL_PATH_1", "/path/to/downloaded-models/google/gemma-2-2b-it")
HF_MODEL_PATH_2 = os.getenv("HF_MODEL_PATH_2", "/path/to/downloaded-models/Qwen/Qwen2.5-3B-Instruct")
HF_MODEL_PATH_3 = os.getenv("HF_MODEL_PATH_3", "/path/to/downloaded-models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
VLLM_MAX_MODEL_LEN = os.getenv("VLLM_MAX_MODEL_LEN")
VLLM_MAX_MODEL_LEN = int(VLLM_MAX_MODEL_LEN) if VLLM_MAX_MODEL_LEN else None
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "auto")
VLLM_TRUST_REMOTE_CODE = _env_bool("VLLM_TRUST_REMOTE_CODE", True)
VLLM_ENFORCE_EAGER = _env_bool("VLLM_ENFORCE_EAGER", False)

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
        "max_tokens": 512,
        "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "dtype": VLLM_DTYPE,
        "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
        "enforce_eager": VLLM_ENFORCE_EAGER,
    },
    {
        "name": "Qwen/Qwen3-8B",
        "provider": "huggingface",
        "path": HF_MODEL_PATH_2,
        "temperature": 1,
        "max_tokens": 512,
        "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "dtype": VLLM_DTYPE,
        "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
        "enforce_eager": VLLM_ENFORCE_EAGER,
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "provider": "huggingface",
        "path": HF_MODEL_PATH_3,
        "temperature": 1,
        "max_tokens": 512,
        "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "dtype": VLLM_DTYPE,
        "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
        "enforce_eager": VLLM_ENFORCE_EAGER,
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

# Model loading strategy:
# - all_at_once: load all configured models first, then evaluate.
# - one_by_one: load/evaluate/analyze/release each model sequentially.
MODEL_LOADING_STRATEGY = os.getenv("MODEL_LOADING_STRATEGY", "all_at_once")
