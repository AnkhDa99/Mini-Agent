from sqlalchemy.orm import Session
from datetime import datetime
from app.models.conversation import Conversation
from app.core.time_utils import now_shanghai


def get_by_uid(db: Session, conversation_uid: str, owner_id: int | None = None):
    q = db.query(Conversation).filter(
        Conversation.conversation_uid == conversation_uid
    )
    if owner_id is not None:
        q = q.filter(Conversation.owner_id == owner_id)
    return q.first()


def create_conversation(
    db: Session,
    conversation_uid: str,
    title: str | None = None,
    owner_id: int | None = None,
):
    obj = Conversation(
        conversation_uid=conversation_uid,
        title=title,
        owner_id=owner_id,
        created_at=now_shanghai(),
        updated_at=now_shanghai(),
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def touch_conversation(db: Session, conversation_id: int):
    obj = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if obj:
        obj.updated_at = now_shanghai()
        db.commit()
    return obj


def update_conversation_title(db: Session, conversation_id: int, title: str):
    obj = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if obj:
        obj.title = title
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj


def update_conversation_summary(db: Session, conversation_id: int, summary: str | None):
    obj = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if obj:
        obj.summary = summary
        obj.updated_at = now_shanghai()
        db.commit()
        db.refresh(obj)

    return obj


def list_recent_conversations(
    db: Session,
    limit: int = 50,
    owner_id: int | None = None,
):
    q = db.query(Conversation)
    if owner_id is not None:
        q = q.filter(Conversation.owner_id == owner_id)
    return (
        q.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
        .all()
    )


from app.models.message import Message


def delete_conversation_and_messages(
    db: Session,
    conversation_uid: str,
    owner_id: int | None = None,
) -> bool:
    q = db.query(Conversation).filter(
        Conversation.conversation_uid == conversation_uid
    )
    if owner_id is not None:
        q = q.filter(Conversation.owner_id == owner_id)
    conversation = q.first()

    if conversation is None:
        return False

    db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).delete()

    db.delete(conversation)
    db.commit()
    return True
