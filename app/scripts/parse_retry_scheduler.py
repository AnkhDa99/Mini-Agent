import time
from app.core.database import SessionLocal
from app.repository.parse_task_repo import (
    recover_timeout_processing_tasks,
    list_pending_parse_tasks,
)
from app.mq.kafka_producer import send_document_parse_message
from app.repository.document_repo import (
    get_document_by_uid,
    reset_document_parse_status,
)

def run_once():
    db = SessionLocal()
    try:
        recovered = recover_timeout_processing_tasks(
            db=db,
            timeout_minutes=10,
        )

        if recovered:
            print(f"[ParseRetryScheduler] recovered processing tasks: {recovered}")

        tasks = list_pending_parse_tasks(db=db, limit=20)

        for task in tasks:
            doc = get_document_by_uid(db, task.document_uid)
            if not doc:
                print(f"[ParseRetryScheduler] document not found: {task.document_uid}")
                continue

            reset_document_parse_status(db, task.document_uid)

            send_document_parse_message(
                {
                    "task_uid": task.task_uid,
                    "document_uid": doc.document_uid,
                    "object_key": doc.object_key,
                }
            )

            print(f"[ParseRetryScheduler] resend parse task: {task.task_uid}")

    finally:
        db.close()


def main():
    print("[ParseRetryScheduler] started")

    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ParseRetryScheduler] error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()