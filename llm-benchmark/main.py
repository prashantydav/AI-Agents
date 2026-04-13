import os
import ast
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from typing import List

load_dotenv()

def safe_parse_metadata(x):
    # Handle NaN
    if pd.isna(x):
        return {}

    # Already dict
    if isinstance(x, dict):
        return x

    # Not string
    if not isinstance(x, str):
        return {}

    x = x.strip()

    # Fix missing braces
    if not x.startswith("{"):
        x = "{" + x
    if not x.endswith("}"):
        x = x + "}"

    # Try JSON
    try:
        return json.loads(x)
    except:
        pass

    # Try Python dict
    try:
        return ast.literal_eval(x)
    except:
        return {}

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
from models.openai import OpenAIModel
from models.huggingface import HuggingFaceModel

# Evaluation
from evaluation.metrics import compute_score
from evaluation.judge import LLMJudge


# ---------------------------
# Load Dataset
# ---------------------------


def enrich_metadata(meta):
    meta.setdefault("evaluation_type", "fuzzy")
    meta.setdefault("category", "qa")
    return meta


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported file format")

    # Ensure metadata is dict
    if isinstance(df.iloc[0]["metadata"], str):
        df["metadata"] = df["metadata"].apply(safe_parse_metadata)
        df["metadata"] = df["metadata"].apply(enrich_metadata)

    return df

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def unwrap_model_output(raw_output):
    if isinstance(raw_output, dict):
        output = raw_output.get("output")
        if output is None:
            return raw_output.get("error", "")
        return str(output).strip()
    if raw_output is None:
        return ""
    return str(raw_output).strip()


def ensure_results_dirs(base_path: str = "results"):
    os.makedirs(base_path, exist_ok=True)
    insights_dir = os.path.join(base_path, "insights")
    plots_dir = os.path.join(insights_dir, "plots")
    os.makedirs(insights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return insights_dir, plots_dir


# ---------------------------
# Initialize Models
# ---------------------------
def init_models() -> List:
    models = []

    for cfg in MODEL_CONFIGS:

        if cfg["provider"] == "ollama":
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
        elif cfg["provider"] == "openai":
            models.append(
                OpenAIModel(
                    api_key=OPENAI_API_KEY,
                    model=cfg["name"]
                )
            )
        elif cfg["provider"] == "huggingface":
            model_path = cfg.get("path", cfg.get("name"))
            models.append(
                HuggingFaceModel(
                    model_path=model_path,
                    model_name=cfg.get("name"),
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
def run_benchmark(df: pd.DataFrame, models: List, judge=None) -> pd.DataFrame:

    results = []

    for model in models:
        print(f"\n🚀 Running model: {model.model_name}")

        for idx, row in df.iterrows():

            raw_prompt = row["problem"]

            prompt = f"""
            Answer the following question.

            STRICT INSTRUCTIONS:
            - Return ONLY the final answer
            - Do NOT explain
            - Do NOT add extra words
            - Do NOT add sentences
            - Output must be short and exact

            Question:
            {raw_prompt}

            Answer:
            """
            ground_truth = row["answer"]
            metadata = row.get("metadata", {})

            eval_type = metadata.get("evaluation_type", "exact_match")

            # ---------------------------
            # Generate Output
            # ---------------------------
            start_time = time.time()
            raw_output = model.generate(prompt)
            output = unwrap_model_output(raw_output)
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



def run_absa_benchmark(df: pd.DataFrame, models: List, judge=None) -> pd.DataFrame:

    results = []

    for model in models:
        print(f"\n🚀 Running model: {model.model_name}")
        print(df.columns)
        for idx, row in df.iterrows():

            text = row["translated_text"]
            aspect = row["Keyword"]
            ground_truth = row["Sentiment"]

            metadata = row.get("metadata", {})
            # eval_type = metadata.get("evaluation_type", "exact_match")
            eval_type = "exact_match"

            # ---------------------------
            # ABSA Prompt
            # ---------------------------
            prompt = f"""
            You are performing Aspect-Based Sentiment Analysis.

            STRICT INSTRUCTIONS:
            - Identify sentiment ONLY for the given aspect
            - Return ONLY one word: Positive / Negative / Neutral
            - Do NOT explain
            - Do NOT add extra text

            Text:
            {text}

            Aspect:
            {aspect}

            Sentiment:
            """

            # ---------------------------
            # Generate Output
            # ---------------------------
            start_time = time.time()
            raw_output = model.generate(prompt)
            output = unwrap_model_output(raw_output)
            latency = time.time() - start_time

            # Normalize output (important)
            # output = output.split("\n")[0]

            # ---------------------------
            # Metric Score
            # ---------------------------
            metric_score = compute_score(
                pred=output,
                gt=ground_truth,
                eval_type=eval_type
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
                "category": metadata.get("category", "absa"),
                "text": text,
                "aspect": aspect,
                "ground_truth": ground_truth,
                "output": output,
                "metric_score": metric_score,
                "judge_score": judge_score,
                "final_score": final_score,
                "latency": latency,
                "judge_reason": judge_reason
            })

            if VERBOSE:
                print(f"✔ Test {idx} | GT: {ground_truth} | Pred: {output} | Score: {final_score:.2f}")

    return pd.DataFrame(results)

# ---------------------------
# Analysis
# ---------------------------
def analyze_results(df: pd.DataFrame):
    insights_dir, plots_dir = ensure_results_dirs("results")

    overall = (
        df.groupby("model", as_index=False)
        .agg(
            avg_final_score=("final_score", "mean"),
            avg_metric_score=("metric_score", "mean"),
            avg_latency=("latency", "mean"),
            samples=("test_id", "count"),
        )
        .sort_values("avg_final_score", ascending=False)
    )

    by_category = (
        df.groupby(["model", "category"], as_index=False)["final_score"]
        .mean()
        .rename(columns={"final_score": "avg_final_score"})
    )

    print("\n📊 Overall Performance:\n")
    print(overall.set_index("model")["avg_final_score"])

    print("\n📊 Category-wise Performance:\n")
    print(by_category.set_index(["model", "category"])["avg_final_score"])

    print("\n⏱ Average Latency:\n")
    print(overall.set_index("model")["avg_latency"])

    overall.to_csv(os.path.join(insights_dir, "overall_summary.csv"), index=False)
    by_category.to_csv(os.path.join(insights_dir, "category_summary.csv"), index=False)

    # Bar chart: avg final score by model
    plt.figure(figsize=(10, 6))
    plt.bar(overall["model"], overall["avg_final_score"])
    plt.title("Average Final Score by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Final Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_final_score_by_model.png"))
    plt.close()

    # Bar chart: avg latency by model
    plt.figure(figsize=(10, 6))
    plt.bar(overall["model"], overall["avg_latency"])
    plt.title("Average Latency by Model")
    plt.xlabel("Model")
    plt.ylabel("Latency (seconds)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_latency_by_model.png"))
    plt.close()

    # Grouped bars per category/model
    pivot = by_category.pivot(index="category", columns="model", values="avg_final_score").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.set_title("Average Final Score by Category and Model")
    ax.set_xlabel("Category")
    ax.set_ylabel("Average Final Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_final_score_by_category.png"))
    plt.close()

    best_model = overall.iloc[0].to_dict() if not overall.empty else {}
    fastest_model = overall.sort_values("avg_latency").iloc[0].to_dict() if not overall.empty else {}

    insights = {
        "best_model_by_final_score": best_model,
        "fastest_model_by_latency": fastest_model,
        "overall_model_summary_file": os.path.join(insights_dir, "overall_summary.csv"),
        "category_summary_file": os.path.join(insights_dir, "category_summary.csv"),
        "plots": {
            "avg_final_score_by_model": os.path.join(plots_dir, "avg_final_score_by_model.png"),
            "avg_latency_by_model": os.path.join(plots_dir, "avg_latency_by_model.png"),
            "avg_final_score_by_category": os.path.join(plots_dir, "avg_final_score_by_category.png"),
        },
    }

    with open(os.path.join(insights_dir, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    with open(os.path.join(insights_dir, "insights.txt"), "w", encoding="utf-8") as f:
        if best_model:
            f.write(
                "Best model by final score: "
                f"{best_model['model']} ({best_model['avg_final_score']:.4f})\n"
            )
        if fastest_model:
            f.write(
                "Fastest model by latency: "
                f"{fastest_model['model']} ({fastest_model['avg_latency']:.4f}s)\n"
            )
        f.write(
            "Saved plots:\n"
            f"- {os.path.join(plots_dir, 'avg_final_score_by_model.png')}\n"
            f"- {os.path.join(plots_dir, 'avg_latency_by_model.png')}\n"
            f"- {os.path.join(plots_dir, 'avg_final_score_by_category.png')}\n"
        )


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
    # results_df = run_benchmark(df, models, judge)
    results_df = run_absa_benchmark(df, models, judge)

    print("💾 Saving results...")

    # Create directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    results_df.to_csv(SAVE_RESULTS_PATH, index=False)

    print("📊 Analyzing results...")
    analyze_results(results_df)

    print("\n✅ Benchmark Completed Successfully!")
