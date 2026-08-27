"""
Embedding 向量缓存（Redis）。
缓存 query → embedding vector，减少 DashScope API 重复调用。

为 Agent 预留接口：
- cache_key_builder: 可自定义 key 生成策略
- invalidate_on_version: 跟随 ES mapping version 自动失效
"""
import hashlib
import json
import logging
from typing import Callable

try:
    import redis
except ImportError:
    redis = None  # type: ignore

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_TTL = 3600  # 1 小时
EMBEDDING_CACHE_PREFIX = "emb:v1"


def _get_redis():
    if redis is None:
        return None
    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        r.ping()
        return r
    except Exception:
        return None


class EmbeddingCache:
    """Embedding 向量缓存层。

    Agent 扩展点:
    - cache_key_builder(query) → str: 自定义 key 生成
    """

    def __init__(self):
        self._redis = _get_redis()
        self.cache_key_builder: Callable[[str], str] | None = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _build_key(self, query: str) -> str:
        if self.cache_key_builder:
            return f"{EMBEDDING_CACHE_PREFIX}:{self.cache_key_builder(query)}"
        digest = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"{EMBEDDING_CACHE_PREFIX}:{digest}"

    def get(self, query: str) -> list[float] | None:
        if not self._redis:
            return None
        try:
            key = self._build_key(query)
            raw = self._redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            logger.debug("EmbeddingCache get failed", exc_info=True)
        return None

    def set(self, query: str, vector: list[float]):
        if not self._redis:
            return
        try:
            key = self._build_key(query)
            self._redis.setex(
                key,
                EMBEDDING_CACHE_TTL,
                json.dumps(vector, ensure_ascii=False),
            )
        except Exception:
            logger.debug("EmbeddingCache set failed", exc_info=True)

    def stats(self) -> dict:
        if not self._redis:
            return {"available": False}
        try:
            info = self._redis.info("keyspace")
            return {"available": True, "keyspace": info}
        except Exception:
            return {"available": True, "error": "stats failed"}


_embedding_cache: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
