from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models.cache_invalidation_task import CacheInvalidationTask
from app.core.time_utils import now_shanghai


TASK_STATUS_PENDING = 0
TASK_STATUS_SUCCESS = 1
TASK_STATUS_RETRY = 2
TASK_STATUS_FAILED = 3


def create_cache_task(
    db: Session,
    task_uid: str,
    cache_key: str,
    conversation_uid: str,
    reason: str,
    status: int = TASK_STATUS_PENDING,
    retry_count: int = 0,
    max_retry: int = 5,
    next_retry_at=None,
    last_error: str | None = None,
):
    obj = CacheInvalidationTask(
        task_uid=task_uid,
        cache_key=cache_key,
        conversation_uid=conversation_uid,
        reason=reason,
        status=status,
        retry_count=retry_count,
        max_retry=max_retry,
        next_retry_at=next_retry_at,
        last_error=last_error,
        created_at=now_shanghai(),
        updated_at=now_shanghai(),
    )

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_task_by_uid(db: Session, task_uid: str):
    return (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.task_uid == task_uid)
        .first()
    )


def list_retryable_tasks(db: Session, limit: int = 50):
    now = now_shanghai()

    return (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.status.in_([TASK_STATUS_PENDING, TASK_STATUS_RETRY]))
        .filter(CacheInvalidationTask.retry_count < CacheInvalidationTask.max_retry)
        .filter(
            CacheInvalidationTask.next_retry_at.is_(None)
            | (CacheInvalidationTask.next_retry_at <= now)
        )
        .order_by(CacheInvalidationTask.created_at.asc())
        .limit(limit)
        .all()
    )


def mark_task_success(db: Session, task_uid: str):
    obj = (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.task_uid == task_uid)
        .first()
    )

    if obj:
        obj.status = TASK_STATUS_SUCCESS
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj


def mark_task_retry(db: Session, task_uid: str, error: str, delay_seconds: int = 30):
    obj = (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.task_uid == task_uid)
        .first()
    )

    if obj:
        obj.retry_count += 1
        obj.status = TASK_STATUS_RETRY
        obj.last_error = error
        obj.next_retry_at = now_shanghai() + timedelta(seconds=delay_seconds)
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj


def mark_task_failed(db: Session, task_uid: str, error: str):
    obj = (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.task_uid == task_uid)
        .first()
    )

    if obj:
        obj.status = TASK_STATUS_FAILED
        obj.last_error = error
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj

def mark_task_failed_with_increment(db: Session, task_uid: str, error: str):
    obj = (
        db.query(CacheInvalidationTask)
        .filter(CacheInvalidationTask.task_uid == task_uid)
        .first()
    )

    if obj:
        obj.retry_count += 1
        obj.status = TASK_STATUS_FAILED
        obj.last_error = error
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj