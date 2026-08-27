import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from app.core.time_utils import now_shanghai

from app.core.redis_client import redis_client
from app.mq.kafka_producer import send_cache_invalidation_message
from app.repository.cache_task_repo import create_cache_task

logger = logging.getLogger(__name__)


def build_chat_context_cache_key(conversation_uid: str) -> str:
    return f"chat:ctx:{conversation_uid}"


def delete_cache_with_retry_task(
    db: Session,
    conversation_uid: str,
    reason: str,
) -> bool:
    """
    删除 Redis 缓存。
    成功：直接返回 True。
    失败：写 cache_invalidation_task，并发送 Kafka 消息异步重试。
    """
    cache_key = build_chat_context_cache_key(conversation_uid)

    try:
        redis_client.delete(cache_key)
        logger.info(
            "Redis cache deleted | conversation_uid=%s cache_key=%s reason=%s",
            conversation_uid,
            cache_key,
            reason,
        )
        return True

    except Exception as exc:
        logger.warning(
            "Redis cache delete failed, create retry task | conversation_uid=%s cache_key=%s error=%s",
            conversation_uid,
            cache_key,
            exc,
        )

        task_uid = f"cache_del_{uuid.uuid4().hex}"

        task = create_cache_task(
            db=db,
            task_uid=task_uid,
            cache_key=cache_key,
            conversation_uid=conversation_uid,
            reason=reason,
            status=0,
            retry_count=0,
            max_retry=5,
            next_retry_at=now_shanghai() + timedelta(seconds=10),
            last_error=str(exc),
        )

        message = {
            "task_uid": task.task_uid,
            "cache_key": task.cache_key,
            "conversation_uid": task.conversation_uid,
            "reason": task.reason,
        }

        try:
            logger.info(
                "准备发送 Kafka 缓存失效消息 | task_uid=%s cache_key=%s conversation_uid=%s reason=%s",
                task.task_uid,
                task.cache_key,
                task.conversation_uid,
                task.reason,
            )

            send_cache_invalidation_message(message)

            logger.info(
                "Kafka 缓存失效消息发送成功 | task_uid=%s cache_key=%s",
                task.task_uid,
                task.cache_key,
            )

        except Exception as mq_exc:
            # 注意：MQ 失败不能影响主业务，因为任务已经落库
            task.last_error = f"redis_error={exc}; kafka_error={mq_exc}"
            db.commit()

            logger.warning(
                "Kafka send failed, task kept in MySQL | task_uid=%s error=%s",
                task.task_uid,
                mq_exc,
            )

        return False