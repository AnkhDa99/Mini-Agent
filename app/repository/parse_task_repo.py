from sqlalchemy.orm import Session

from app.models.document_parse_task import DocumentParseTask
from app.core.time_utils import now_shanghai


def get_parse_task_by_uid(db: Session, task_uid: str):
    return (
        db.query(DocumentParseTask)
        .filter(DocumentParseTask.task_uid == task_uid)
        .first()
    )


def mark_parse_task_processing(
    db: Session,
    task_uid: str,
    locked_by: str,
) -> bool:
    affected = (
        db.query(DocumentParseTask)
        .filter(
            DocumentParseTask.task_uid == task_uid,
            DocumentParseTask.status.in_(["pending", "retry"]),
        )
        .update(
            {
                DocumentParseTask.status: "processing",
                DocumentParseTask.locked_by: locked_by,
                DocumentParseTask.locked_at: now_shanghai(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return affected == 1


def mark_parse_task_success(db: Session, task_uid: str):
    task = get_parse_task_by_uid(db, task_uid)
    if task:
        task.status = "success"
        task.last_error = None
        db.commit()
        db.refresh(task)
    return task


def mark_parse_task_failed(db: Session, task_uid: str, error: str):
    task = get_parse_task_by_uid(db, task_uid)
    if task:
        task.status = "failed"
        task.last_error = error
        db.commit()
        db.refresh(task)
    return task

def list_pending_parse_tasks(db: Session, limit: int = 20):
    return (
        db.query(DocumentParseTask)
        .filter(DocumentParseTask.status.in_(["pending", "retry"]))
        .order_by(DocumentParseTask.id.asc())
        .limit(limit)
        .all()
    )

def mark_parse_task_retry_or_failed(db: Session, task_uid: str, error: str):
    task = get_parse_task_by_uid(db, task_uid)
    if not task:
        return None

    task.retry_count += 1
    task.last_error = error

    if task.retry_count >= task.max_retry:
        task.status = "failed"
    else:
        task.status = "retry"
        task.next_retry_at = now_shanghai() + timedelta(minutes=2)

    task.locked_by = None
    task.locked_at = None

    db.commit()
    db.refresh(task)
    return task

from datetime import timedelta

def recover_timeout_processing_tasks(db: Session, timeout_minutes: int = 10):
    deadline = now_shanghai() - timedelta(minutes=timeout_minutes)

    affected = (
        db.query(DocumentParseTask)
        .filter(
            DocumentParseTask.status == "processing",
            DocumentParseTask.locked_at < deadline,
        )
        .update(
            {
                DocumentParseTask.status: "retry",
                DocumentParseTask.locked_by: None,
                DocumentParseTask.locked_at: None,
                DocumentParseTask.last_error: "processing timeout recovered",
                DocumentParseTask.next_retry_at: now_shanghai(),
            },
            synchronize_session=False,
        )
    )

    db.commit()
    return affected