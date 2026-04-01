import time
from openai import OpenAI

class OpenAIModel:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.name = "GPT-4o"

        # Pricing (approx, update if needed)
        self.input_cost_per_1k = 0.00015   # $ per 1K tokens
        self.output_cost_per_1k = 0.0006

    def generate(self, prompt, temperature=0.0, max_tokens=500):
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            output = response.choices[0].message.content.strip()

            # Token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            # Cost calculation
            cost = (
                (input_tokens / 1000) * self.input_cost_per_1k +
                (output_tokens / 1000) * self.output_cost_per_1k
            )

            latency = time.time() - start_time

            return {
                "output": output,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "error": None
            }

        except Exception as e:
            return {
                "output": None,
                "latency": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0,
                "error": str(e)
            }