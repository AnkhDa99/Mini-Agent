from sqlalchemy.orm import Session
from datetime import datetime
from app.models.message import Message
from app.core.time_utils import now_shanghai

def get_message_by_id(db: Session, message_id: int) -> Message | None:
    return db.query(Message).filter(Message.id == message_id).first()


def create_message(db: Session, conversation_id: int, role: str, content: str, token_count: int | None = None, sources_json: str | None = None, metadata_json: str | None = None):
    obj = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        token_count=token_count,
        sources_json=sources_json,
        metadata_json=metadata_json,
        created_at=now_shanghai(),
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_messages_by_conversation(db: Session, conversation_id: int):
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

def list_recent_messages_by_conversation(db: Session, conversation_id: int, limit: int = 20):
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def count_messages_by_conversation(db: Session, conversation_id: int) -> int:
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .count()
    )

def delete_messages_by_ids(db: Session, message_ids: list[int]) -> int:
    if not message_ids:
        return 0

    rows = (
        db.query(Message)
        .filter(Message.id.in_(message_ids))
        .all()
    )

    count = len(rows)

    for row in rows:
        db.delete(row)

    db.commit()
    return count

def get_last_assistant_message(db: Session, conversation_id: int):
    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.role == "assistant",
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .first()
    )

def get_last_user_message(db: Session, conversation_id: int):
    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.role == "user",
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .first()
    )

def update_message_content(db: Session, message_id: int, content: str):
    obj = db.query(Message).filter(Message.id == message_id).first()

    if obj:
        obj.content = content
        db.commit()
        db.refresh(obj)

    return obj

def delete_messages_after_id(db: Session, conversation_id: int, message_id: int) -> int:
    rows = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.id > message_id,
        )
        .all()
    )

    count = len(rows)

    for row in rows:
        db.delete(row)

    db.commit()
    return count