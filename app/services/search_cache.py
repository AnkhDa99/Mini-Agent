"""
检索结果缓存（Redis）。
缓存 query → [chunk_uid, ...] 的检索结果，高频查询直接返回。

与 EmbeddingCache 配合：embedding 缓存消除 API 调用，search 缓存消除 FAISS+ES 检索。
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

SEARCH_CACHE_TTL = 1800  # 30 分钟
SEARCH_CACHE_PREFIX = "search:v2"


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


class SearchCache:
    """检索结果缓存层。

    Agent 扩展点:
    - cache_key_builder(query, strategy) → str: 考虑检索策略的 key
    - should_cache(result) → bool: Agent 判断是否值得缓存
    """

    def __init__(self):
        self._redis = _get_redis()
        self.cache_key_builder: Callable[[str, str], str] | None = None
        self.should_cache_hook: Callable[[list[dict]], bool] | None = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _build_key(self, query: str, strategy: str = "default", owner_id: int | None = None) -> str:
        if self.cache_key_builder:
            return f"{SEARCH_CACHE_PREFIX}:{self.cache_key_builder(query, strategy)}"
        suffix = f"|uid={owner_id}" if owner_id is not None else ""
        digest = hashlib.md5(f"{query.strip().lower()}|{strategy}{suffix}".encode()).hexdigest()[:16]
        return f"{SEARCH_CACHE_PREFIX}:{digest}"

    def get(self, query: str, strategy: str = "default", owner_id: int | None = None) -> list[str] | None:
        """返回缓存的 chunk_uid 列表。"""
        if not self._redis:
            return None
        try:
            key = self._build_key(query, strategy, owner_id=owner_id)
            raw = self._redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            logger.debug("SearchCache get failed", exc_info=True)
        return None

    def set(self, query: str, chunk_uids: list[str], strategy: str = "default", owner_id: int | None = None):
        if not self._redis:
            return
        if self.should_cache_hook and not self.should_cache_hook(chunk_uids):
            return
        try:
            key = self._build_key(query, strategy, owner_id=owner_id)
            self._redis.setex(key, SEARCH_CACHE_TTL, json.dumps(chunk_uids))
        except Exception:
            logger.debug("SearchCache set failed", exc_info=True)

    def stats(self) -> dict:
        if not self._redis:
            return {"available": False}
        try:
            return {"available": True}
        except Exception:
            return {"available": False}


_search_cache: SearchCache | None = None


def get_search_cache() -> SearchCache:
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchCache()
    return _search_cache
