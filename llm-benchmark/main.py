import pandas as pd
import time
import json

from typing import List

# Config
from config import (
    MODEL_CONFIGS,
    JUDGE_MODEL,
    OLLAMA_BASE_URL,
    DATASET_PATH,
    USE_JUDGE,
    SAVE_RESULTS_PATH,
    REQUEST_TIMEOUT,
    RETRY_COUNT,
    VERBOSE
)

# Model
from models.ollama import OllamaModel

# Evaluation
from evaluation.metrics import compute_score
from evaluation.judge import LLMJudge


# ---------------------------
# Load Dataset
# ---------------------------
def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported file format")

    # Ensure metadata is dict
    if isinstance(df.iloc[0]["metadata"], str):
        df["metadata"] = df["metadata"].apply(json.loads)

    return df


# ---------------------------
# Initialize Models
# ---------------------------
def init_models() -> List[OllamaModel]:
    models = []

    for cfg in MODEL_CONFIGS:
        models.append(
            OllamaModel(
                model_name=cfg["name"],
                base_url=OLLAMA_BASE_URL,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                timeout=REQUEST_TIMEOUT,
                retry_count=RETRY_COUNT
            )
        )

    return models


# ---------------------------
# Initialize Judge
# ---------------------------
def init_judge():
    if not USE_JUDGE:
        return None

    judge_model = OllamaModel(
        model_name=JUDGE_MODEL["name"],
        base_url=OLLAMA_BASE_URL,
        temperature=JUDGE_MODEL["temperature"],
        max_tokens=JUDGE_MODEL["max_tokens"],
        timeout=REQUEST_TIMEOUT,
        retry_count=RETRY_COUNT
    )

    return LLMJudge(client=judge_model, model_name="judge")


# ---------------------------
# Benchmark Runner
# ---------------------------
def run_benchmark(df: pd.DataFrame, models: List[OllamaModel], judge=None) -> pd.DataFrame:

    results = []

    for model in models:
        print(f"\n🚀 Running model: {model.model_name}")

        for idx, row in df.iterrows():

            prompt = row["problem"]
            ground_truth = row["answer"]
            metadata = row.get("metadata", {})

            eval_type = metadata.get("evaluation_type", "exact_match")

            # ---------------------------
            # Generate Output
            # ---------------------------
            start_time = time.time()
            output = model.generate(prompt)
            latency = time.time() - start_time

            # ---------------------------
            # Metric Score
            # ---------------------------
            metric_score = compute_score(
                pred=output,
                gt=ground_truth,
                eval_type=eval_type,
                **metadata
            )

            # ---------------------------
            # Judge Score
            # ---------------------------
            judge_score = None
            judge_reason = None

            if judge:
                judge_result = judge.judge(prompt, output, ground_truth)
                judge_score = judge_result.get("score", 0)
                judge_reason = judge_result.get("reason", "")

            # ---------------------------
            # Final Score
            # ---------------------------
            if judge_score is not None:
                final_score = 0.5 * metric_score + 0.5 * (judge_score / 5)
            else:
                final_score = metric_score

            # ---------------------------
            # Store Result
            # ---------------------------
            results.append({
                "model": model.model_name,
                "test_id": idx,
                "category": metadata.get("category", "unknown"),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "output": output,
                "metric_score": metric_score,
                "judge_score": judge_score,
                "final_score": final_score,
                "latency": latency,
                "judge_reason": judge_reason
            })

            if VERBOSE:
                print(f"✔ Test {idx} | Score: {final_score:.2f}")

    return pd.DataFrame(results)


# ---------------------------
# Analysis
# ---------------------------
def analyze_results(df: pd.DataFrame):

    print("\n📊 Overall Performance:\n")
    print(df.groupby("model")["final_score"].mean())

    print("\n📊 Category-wise Performance:\n")
    print(df.groupby(["model", "category"])["final_score"].mean())

    print("\n⏱ Average Latency:\n")
    print(df.groupby("model")["latency"].mean())


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":

    print("📂 Loading dataset...")
    df = load_dataset(DATASET_PATH)
    df = df.head(25)

    print("🤖 Initializing models...")
    models = init_models()

    print("⚖️ Initializing judge...")
    judge = init_judge()

    print("🚀 Running benchmark...")
    results_df = run_benchmark(df, models, judge)

    print("💾 Saving results...")
    results_df.to_csv(SAVE_RESULTS_PATH, index=False)

    print("📊 Analyzing results...")
    analyze_results(results_df)

    print("\n✅ Benchmark Completed Successfully!")