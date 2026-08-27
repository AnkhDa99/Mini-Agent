import json
from datetime import datetime

from app.core.config import settings
from app.core.redis_client import redis_client

CTX_KEY = "chat:ctx:{}"
RECENT_KEY = "chat:recent"


def _serialize_message(message: dict) -> str:
    return json.dumps(message, ensure_ascii=False)


def _deserialize_message(raw: str) -> dict:
    return json.loads(raw)


def get_cached_context_messages(conversation_uid: str) -> list[dict] | None:
    key = CTX_KEY.format(conversation_uid)
    rows = redis_client.lrange(key, 0, -1)
    if not rows:
        return None
    return [_deserialize_message(row) for row in rows]


def append_context_message(conversation_uid: str, role: str, content: str, created_at: datetime | None = None):
    key = CTX_KEY.format(conversation_uid)
    payload = {
        "role": role,
        "content": content,
        "created_at": (created_at or datetime.utcnow()).isoformat(),
    }
    redis_client.rpush(key, _serialize_message(payload))
    redis_client.ltrim(key, -settings.chat_context_limit, -1)
    redis_client.expire(key, settings.chat_cache_ttl_seconds)


def set_context_messages(conversation_uid: str, messages: list[dict]):
    key = CTX_KEY.format(conversation_uid)
    pipe = redis_client.pipeline()
    pipe.delete(key)
    if messages:
        for msg in messages[-settings.chat_context_limit:]:
            pipe.rpush(key, _serialize_message(msg))
        pipe.expire(key, settings.chat_cache_ttl_seconds)
    pipe.execute()


def delete_context_messages(conversation_uid: str):
    key = CTX_KEY.format(conversation_uid)
    redis_client.delete(key)


def touch_recent_conversation(conversation_uid: str, title: str | None, updated_at: datetime | None = None):
    ts = (updated_at or datetime.utcnow()).timestamp()
    redis_client.zadd(RECENT_KEY, {conversation_uid: ts})


def remove_recent_conversation(conversation_uid: str):
    redis_client.zrem(RECENT_KEY, conversation_uid)


def get_recent_conversation_uids(limit: int = 20) -> list[str]:
    rows = redis_client.zrevrange(RECENT_KEY, 0, limit - 1)
    return rows or []

def remove_recent_conversation(conversation_uid: str):
    redis_client.zrem(RECENT_KEY, conversation_uid)

def delete_conversation_cache(conversation_uid: str):
    delete_context_messages(conversation_uid)
    remove_recent_conversation(conversation_uid)

def touch_recent_conversation(conversation_uid: str, title: str | None, updated_at: datetime | None = None):
    ts = (updated_at or datetime.utcnow()).timestamp()
    redis_client.zadd(RECENT_KEY, {conversation_uid: ts})

    total = redis_client.zcard(RECENT_KEY)
    keep_n = settings.chat_recent_conversation_limit

    if total > keep_n:
        redis_client.zremrangebyrank(RECENT_KEY, 0, total - keep_n - 1)

