from __future__ import annotations

import json
import logging
from hashlib import sha256
from typing import Any, Optional

try:
    import redis
except ImportError:
    redis = None

from config import REDIS_URL

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str = REDIS_URL):
        self.url = url
        self.client = None

        if redis is None:
            logger.warning("redis package is not installed; Redis caching is disabled.")
            return

        try:
            self.client = redis.Redis.from_url(url, decode_responses=True)
            self.client.ping()
        except Exception as exc:
            logger.warning("Redis is unavailable at %s; caching disabled. Error: %s", url, exc)
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def build_key(self, prefix: str, *parts: str) -> str:
        normalized = "::".join(str(part) for part in parts)
        digest = sha256(normalized.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def get_json(self, key: str) -> Optional[Any]:
        if not self.client:
            return None

        payload = self.client.get(key)
        if payload is None:
            return None

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not self.client:
            return
        self.client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
