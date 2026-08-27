from sqlalchemy import Column, BigInteger, String, DateTime, Text, ForeignKey

from app.core.database import Base
from app.core.time_utils import now_shanghai


class Conversation(Base):
    __tablename__ = "conversation"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_uid = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True, index=True)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)