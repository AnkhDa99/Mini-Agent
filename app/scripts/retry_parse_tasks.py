from app.core.database import SessionLocal
from app.repository.parse_task_repo import (
    list_pending_parse_tasks,
    recover_timeout_processing_tasks,
)
from app.repository.document_repo import get_document_by_uid
from app.mq.kafka_producer import send_document_parse_message


def main():
    db = SessionLocal()
    try:
        recovered = recover_timeout_processing_tasks(db, timeout_minutes=10)
        print(f"recovered processing tasks: {recovered}")

        tasks = list_pending_parse_tasks(db, limit=20)

        for task in tasks:
            doc = get_document_by_uid(db, task.document_uid)
            if not doc:
                continue

            send_document_parse_message(
                {
                    "task_uid": task.task_uid,
                    "document_uid": doc.document_uid,
                    "object_key": doc.object_key,
                }
            )

            print(f"resend parse task: {task.task_uid}")

    finally:
        db.close()


if __name__ == "__main__":
    main()