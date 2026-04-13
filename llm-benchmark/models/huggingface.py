import os
import time
import gc

from vllm import LLM, SamplingParams


class HuggingFaceModel:
    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
        retry_count: int = 2,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
    ):
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(os.path.normpath(model_path))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager

        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        self.llm = LLM(
            **llm_kwargs
        )

    def generate(self, prompt: str) -> str:
        error_msg = "Unknown error"

        for attempt in range(self.retry_count + 1):
            try:
                sampling_params = SamplingParams(
                    temperature=max(self.temperature, 0.0),
                    max_tokens=self.max_tokens,
                )

                start_time = time.time()
                outputs = self.llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
                _ = time.time() - start_time  # kept for consistency with other wrappers

                output_text = ""
                if outputs and outputs[0].outputs:
                    output_text = outputs[0].outputs[0].text.strip()

                if not output_text:
                    return "ERROR: Empty response from vLLM model"
                return output_text
            except Exception as e:
                error_msg = str(e)
                if attempt < self.retry_count:
                    time.sleep(1)
                else:
                    return f"ERROR: {error_msg}"

        return f"ERROR: {error_msg}"

    def unload(self):
        if hasattr(self, "llm"):
            del self.llm

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        gc.collect()
