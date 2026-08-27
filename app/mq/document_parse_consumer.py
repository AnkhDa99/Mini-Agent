import json
import socket
import uuid
import logging
import threading

from kafka import KafkaConsumer

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.parse_service import ParseService


logger = logging.getLogger(__name__)

_consumer_thread = None
_stop_event = threading.Event()


def _run_consumer_loop():
    worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    logger.info(
        "Document parse consumer starting | bootstrap=%s topic=%s worker_id=%s",
        settings.kafka_bootstrap_servers,
        settings.kafka_document_parse_topic,
        worker_id,
    )

    try:
        consumer = KafkaConsumer(
            settings.kafka_document_parse_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id="mini-agent-document-parse-group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=10000,  # 10s 超时，允许定期检查 stop_event
        )
    except Exception as exc:
        logger.warning("Kafka consumer 初始化失败: %s，解析任务将仅由 parse_worker 兜底", exc)
        return

    service = ParseService()

    try:
        while not _stop_event.is_set():
            for msg in consumer:
                if _stop_event.is_set():
                    break

                payload = msg.value
                logger.info("收到解析任务: %s", payload)

                task_uid = payload.get("task_uid")
                document_uid = payload.get("document_uid")

                if not task_uid or not document_uid:
                    logger.warning("解析任务消息缺少字段: %s", payload)
                    continue

                db = SessionLocal()
                try:
                    service.parse_document(
                        db=db,
                        task_uid=task_uid,
                        document_uid=document_uid,
                        worker_id=worker_id,
                    )
                except Exception as exc:
                    logger.exception("解析任务失败: %s", exc)
                finally:
                    db.close()
    finally:
        consumer.close()
        logger.info("Document parse consumer stopped")


def start_document_parse_consumer():
    global _consumer_thread

    if _consumer_thread is not None and _consumer_thread.is_alive():
        logger.info("Document parse consumer already running")
        return

    _stop_event.clear()
    _consumer_thread = threading.Thread(
        target=_run_consumer_loop,
        name="doc-parse-consumer",
        daemon=True,
    )
    _consumer_thread.start()
    logger.info("Document parse consumer thread started")


def stop_document_parse_consumer():
    """通知 Kafka consumer 退出并等待线程结束。"""
    _stop_event.set()
    if _consumer_thread is not None and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=12)
        logger.info("Document parse consumer stopped")
