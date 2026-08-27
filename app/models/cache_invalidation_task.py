from datetime import datetime

from sqlalchemy import Column, BigInteger, String, DateTime, Text, Integer, SmallInteger

from app.core.database import Base
from app.core.time_utils import now_shanghai

class CacheInvalidationTask(Base):
    __tablename__ = "cache_invalidation_task"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    task_uid = Column(String(64), nullable=False, unique=True, index=True)
    cache_key = Column(String(255), nullable=False)
    conversation_uid = Column(String(64), nullable=False, index=True)
    reason = Column(String(64), nullable=False)

    # 0 pending, 1 success, 2 retry, 3 failed
    status = Column(SmallInteger, nullable=False, default=0)

    retry_count = Column(Integer, nullable=False, default=0)
    max_retry = Column(Integer, nullable=False, default=5)
    next_retry_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)