import time
import google.generativeai as genai

class GeminiModel:
    def __init__(self, api_key, model="gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.name = "Gemini-1.5"

        self.input_cost_per_1k = 0.0005
        self.output_cost_per_1k = 0.0015

    def estimate_tokens(self, text):
        return len(text.split()) * 1.3  # rough estimate

    def generate(self, prompt, temperature=0.0, max_tokens=500):
        start_time = time.time()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

            output = response.text.strip()

            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(output)

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