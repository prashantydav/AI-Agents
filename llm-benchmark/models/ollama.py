import requests
import time


class OllamaModel:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
        retry_count: int = 2
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }

        for attempt in range(self.retry_count + 1):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"

            except Exception as e:
                error_msg = str(e)

            # Retry logic
            if attempt < self.retry_count:
                time.sleep(1)
            else:
                return f"ERROR: {error_msg}"

        return ""