import json
import logging
from typing import Any

from kafka import KafkaProducer

from app.core.config import settings

logger = logging.getLogger(__name__)

_producer: KafkaProducer | None = None


def get_kafka_producer() -> KafkaProducer:
    global _producer

    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            retries=3,
            acks="all",
        )

    return _producer


def send_cache_invalidation_message(message: dict[str, Any]) -> None:
    producer = get_kafka_producer()

    future = producer.send(
        topic=settings.kafka_cache_invalidation_topic,
        key=message.get("conversation_uid"),
        value=message,
    )

    result = future.get(timeout=10)

    logger.info(
        "Kafka cache invalidation message sent | topic=%s partition=%s offset=%s task_uid=%s",
        result.topic,
        result.partition,
        result.offset,
        message.get("task_uid"),
    )

def send_document_parse_message(payload: dict):
    producer = get_kafka_producer()
    if producer is None:
        raise RuntimeError("Kafka producer 未初始化")

    topic = settings.kafka_document_parse_topic

    future = producer.send(topic, payload)
    record_metadata = future.get(timeout=10)

    return {
        "topic": record_metadata.topic,
        "partition": record_metadata.partition,
        "offset": record_metadata.offset,
    }