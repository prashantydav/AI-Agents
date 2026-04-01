import json
from typing import Dict

# You can swap this with your OpenAI / Claude wrapper
class LLMJudge:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def _build_prompt(self, prompt: str, prediction: str, ground_truth: str) -> str:
        return f"""
You are an expert evaluator for LLM outputs.

Evaluate the model response based on correctness, completeness, and relevance.

### Original Task:
{prompt}

### Ground Truth:
{ground_truth}

### Model Response:
{prediction}

### Instructions:
- Score from 1 to 5
    1 = completely incorrect
    3 = partially correct
    5 = fully correct

- Be strict but fair
- Penalize hallucinations and wrong facts

### Output Format (STRICT JSON):
{{
  "score": <int>,
  "reason": "<brief explanation>"
}}
"""

    def judge(self, prompt: str, prediction: str, ground_truth: str) -> Dict:
        full_prompt = self._build_prompt(prompt, prediction, ground_truth)

        response = self.client.generate(full_prompt)  # unified interface

        try:
            parsed = json.loads(response)
            return {
                "score": parsed.get("score", 0),
                "reason": parsed.get("reason", "")
            }
        except Exception:
            return {
                "score": 0,
                "reason": "Invalid judge response"
            }