import json
import logging
from datetime import datetime, timedelta

from kafka import KafkaConsumer

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.redis_client import redis_client
from app.repository.cache_task_repo import (
    get_task_by_uid,
    mark_task_failed,
    mark_task_retry,
    mark_task_success,
)

logger = logging.getLogger(__name__)

import time


def delete_cache_with_short_retry(cache_key: str, max_attempts: int = 3):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            redis_client.delete(cache_key)
            return True, None
        except Exception as exc:
            last_error = exc
            time.sleep(attempt)

    return False, last_error

def handle_cache_invalidation_message(message: dict):
    task_uid = message["task_uid"]
    cache_key = message["cache_key"]

    db = SessionLocal()

    try:
        task = get_task_by_uid(db, task_uid)

        if task is None:
            logger.warning("Cache invalidation task not found | task_uid=%s", task_uid)
            return

        if task.status == 1:
            logger.info("Cache invalidation task already success | task_uid=%s", task_uid)
            return

        try:
            ok, error = delete_cache_with_short_retry(cache_key)

            if ok:
                mark_task_success(db, task_uid)
            else:
                if task.retry_count + 1 >= task.max_retry:
                    mark_task_failed(db, task_uid, str(error))
                else:
                    mark_task_retry(
                        db=db,
                        task_uid=task_uid,
                        error=str(error),
                        delay_seconds=30,
                    )

            logger.info(
                "Cache invalidation success | task_uid=%s cache_key=%s",
                task_uid,
                cache_key,
            )

        except Exception as exc:
            next_retry_count = task.retry_count + 1

            if next_retry_count >= task.max_retry:
                mark_task_failed(db, task_uid, str(exc))

                logger.error(
                    "Cache invalidation failed permanently | task_uid=%s error=%s",
                    task_uid,
                    exc,
                )
            else:
                mark_task_retry(
                    db=db,
                    task_uid=task_uid,
                    error=str(exc),
                    delay_seconds=30,
                )

                logger.warning(
                    "Cache invalidation retry later | task_uid=%s retry_count=%s error=%s",
                    task_uid,
                    next_retry_count,
                    exc,
                )

    finally:
        db.close()


def main():
    logger.info(
        "Kafka consumer config | bootstrap=%s topic=%s group=%s",
        settings.kafka_bootstrap_servers,
        settings.kafka_cache_invalidation_topic,
        settings.kafka_consumer_group,
    )
    consumer = KafkaConsumer(
        settings.kafka_cache_invalidation_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    logger.info(
        "Cache invalidation consumer started | topic=%s group=%s",
        settings.kafka_cache_invalidation_topic,
        settings.kafka_consumer_group,
    )

    for record in consumer:
        print("【Kafka Consumer】收到原始消息:", record)

        logger.info(
            "Kafka message received | topic=%s partition=%s offset=%s key=%s value=%s",
            record.topic,
            record.partition,
            record.offset,
            record.key,
            record.value,
        )

        handle_cache_invalidation_message(record.value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()