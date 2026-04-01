import json
import math
import re
from typing import List, Dict
from difflib import SequenceMatcher

# ---------------------------
# Exact Match
# ---------------------------
def exact_match(pred: str, gt: str) -> int:
    if pred is None or gt is None:
        return 0
    return int(pred.strip() == gt.strip())


# ---------------------------
# Normalized Exact Match
# ---------------------------
def normalized_match(pred: str, gt: str) -> int:
    def normalize(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    return int(normalize(pred) == normalize(gt))


# ---------------------------
# Fuzzy Match (Sequence Similarity)
# ---------------------------
def fuzzy_score(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    return SequenceMatcher(None, pred, gt).ratio()


# ---------------------------
# Token Overlap Score (Jaccard)
# ---------------------------
def jaccard_similarity(pred: str, gt: str) -> float:
    pred_tokens = set(pred.lower().split())
    gt_tokens = set(gt.lower().split())
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    intersection = pred_tokens.intersection(gt_tokens)
    union = pred_tokens.union(gt_tokens)
    
    return len(intersection) / len(union)


# ---------------------------
# JSON Validation
# ---------------------------
def validate_json(output: str) -> int:
    try:
        json.loads(output)
        return 1
    except Exception:
        return 0


# ---------------------------
# JSON Schema Match (Optional)
# ---------------------------
def json_key_match(output: str, required_keys: List[str]) -> float:
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, dict):
            return 0.0
        
        matched = sum(1 for key in required_keys if key in parsed)
        return matched / len(required_keys)
    except:
        return 0.0


# ---------------------------
# Code Execution Evaluation
# ---------------------------
def run_code_and_test(code: str, test_cases: List[Dict]) -> float:
    """
    test_cases format:
    [
        {"input": 2, "output": 4},
        {"input": 3, "output": 9}
    ]
    """
    passed = 0
    
    for test in test_cases:
        try:
            local_env = {}
            exec(code, {}, local_env)

            func = list(local_env.values())[0]

            result = func(test["input"])
            if result == test["output"]:
                passed += 1

        except Exception:
            continue

    return passed / len(test_cases) if test_cases else 0.0


# ---------------------------
# Length Penalty (for verbosity control)
# ---------------------------
def length_penalty(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    
    return min(len(gt) / len(pred), 1.0)


# ---------------------------
# Aggregate Score Dispatcher
# ---------------------------
def compute_score(pred: str, gt: str, eval_type: str, **kwargs) -> float:
    
    if eval_type == "exact_match":
        return exact_match(pred, gt)

    elif eval_type == "normalized_match":
        return normalized_match(pred, gt)

    elif eval_type == "fuzzy":
        return fuzzy_score(pred, gt)

    elif eval_type == "jaccard":
        return jaccard_similarity(pred, gt)

    elif eval_type == "json_valid":
        return validate_json(pred)

    elif eval_type == "json_schema":
        return json_key_match(pred, kwargs.get("required_keys", []))

    elif eval_type == "code":
        return run_code_and_test(pred, kwargs.get("test_cases", []))

    else:
        return 0.0