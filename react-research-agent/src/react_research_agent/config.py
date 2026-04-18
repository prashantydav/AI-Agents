from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_steps: int = int(os.getenv("MAX_STEPS", "10"))
    request_timeout_s: int = int(os.getenv("REQUEST_TIMEOUT_S", "20"))
    max_tool_retries: int = int(os.getenv("MAX_TOOL_RETRIES", "2"))
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    max_observation_chars: int = int(os.getenv("MAX_OBSERVATION_CHARS", "2800"))
