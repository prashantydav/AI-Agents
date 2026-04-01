import time
import anthropic

class AnthropicModel:
    def __init__(self, api_key, model="claude-3-5-sonnet-20240620"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = "Claude-3.5"

        self.input_cost_per_1k = 0.003
        self.output_cost_per_1k = 0.015

    def generate(self, prompt, temperature=0.0, max_tokens=500):
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            output = response.content[0].text.strip()

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

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