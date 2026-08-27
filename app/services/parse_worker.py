import logging
import socket
import uuid

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.time_utils import now_shanghai
from app.repository.parse_task_repo import (
    list_pending_parse_tasks,
    recover_timeout_processing_tasks,
)
from app.services.parse_service import ParseService

logger = logging.getLogger(__name__)

_scheduler = None
_parse_service = None
_worker_id = None


def _get_parse_service():
    global _parse_service
    if _parse_service is None:
        _parse_service = ParseService()
    return _parse_service


def _get_worker_id():
    global _worker_id
    if _worker_id is None:
        _worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    return _worker_id


def process_parse_batch():
    """兜底：轮询 DB 中 pending/retry 的解析任务。

    Kafka MQ 是主路径（低延迟），此 worker 仅在 MQ 不可用或消息丢失时兜底。
    同时负责恢复超时的 processing 任务。"""
    db = SessionLocal()
    try:
        recover_timeout_processing_tasks(db, timeout_minutes=10)

        tasks = list_pending_parse_tasks(db, limit=settings.parse_worker_batch_size)
        if not tasks:
            return

        now = now_shanghai()
        valid_tasks = [
            t for t in tasks
            if t.status == "pending"
            or (t.status == "retry" and t.next_retry_at is not None and t.next_retry_at <= now)
        ]

        if not valid_tasks:
            return

        logger.info("Parse worker: processing %d tasks", len(valid_tasks))
        svc = _get_parse_service()
        worker_id = _get_worker_id()

        for task in valid_tasks:
            try:
                svc.parse_document(
                    db=db,
                    task_uid=task.task_uid,
                    document_uid=task.document_uid,
                    worker_id=worker_id,
                )
            except Exception:
                logger.exception("Parse task failed: %s", task.task_uid)

    except Exception:
        logger.exception("Parse worker batch failed")
    finally:
        db.close()


def start_parse_worker():
    global _scheduler

    if _scheduler is not None:
        logger.info("Parse worker already running")
        return

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # Parse 是兜底路径，每 30s 轮询即可（主要靠 Kafka MQ 驱动）
    scheduler.add_job(
        process_parse_batch,
        trigger="interval",
        seconds=30,
        id="parse_worker_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    _scheduler = scheduler

    logger.info("Parse worker started | interval=30s batch_size=%d",
                settings.parse_worker_batch_size)


def stop_parse_worker():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Parse worker stopped")
