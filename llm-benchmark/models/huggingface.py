import os
import time
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceModel:
    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
        retry_count: int = 2,
    ):
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(os.path.normpath(model_path))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        error_msg = "Unknown error"

        for attempt in range(self.retry_count + 1):
            try:
                encoded = self.tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                generation_config = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": self.max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }

                if self.temperature and self.temperature > 0:
                    generation_config["do_sample"] = True
                    generation_config["temperature"] = self.temperature
                else:
                    generation_config["do_sample"] = False

                start_time = time.time()
                with torch.no_grad():
                    output_ids = self.model.generate(**generation_config)

                new_tokens = output_ids[0][input_ids.shape[1] :]
                output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if not output_text:
                    return "ERROR: Empty response from local Hugging Face model"

                _ = time.time() - start_time  # kept for consistency with other wrappers
                return output_text
            except Exception as e:
                error_msg = str(e)
                if attempt < self.retry_count:
                    time.sleep(1)
                else:
                    return f"ERROR: {error_msg}"

        return f"ERROR: {error_msg}"

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        gc.collect()
