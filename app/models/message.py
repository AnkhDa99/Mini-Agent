from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, Text, Integer
from datetime import datetime
from app.core.database import Base
from app.core.time_utils import now_shanghai

class Message(Base):
    __tablename__ = "message"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversation.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    sources_json = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)  # Issue 2: web/tool/general usage metadata
    created_at = Column(DateTime, nullable=False, default=now_shanghai)