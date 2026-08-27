from sqlalchemy.orm import Session

from app.models.user import User


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def create_user(
    db: Session,
    username: str,
    password_hash: str,
    role: str = "user",
    question_limit: int = 0,
    document_limit: int = 0,
) -> User:
    user = User(
        username=username,
        password_hash=password_hash,
        role=role,
        question_limit=question_limit,
        document_limit=document_limit,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def count_users(db: Session) -> int:
    return db.query(User).count()
