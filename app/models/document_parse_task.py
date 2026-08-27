from sqlalchemy import Column, BigInteger, String, DateTime, Integer, Text
from app.core.database import Base
from app.core.time_utils import now_shanghai


PARSE_TASK_PENDING = "pending"
PARSE_TASK_PROCESSING = "processing"
PARSE_TASK_SUCCESS = "success"
PARSE_TASK_RETRY = "retry"
PARSE_TASK_FAILED = "failed"


class DocumentParseTask(Base):
    __tablename__ = "document_parse_task"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    task_uid = Column(String(64), unique=True, nullable=False, index=True)
    document_uid = Column(String(64), nullable=False, index=True)

    status = Column(String(32), nullable=False, default=PARSE_TASK_PENDING)

    retry_count = Column(Integer, nullable=False, default=0)
    max_retry = Column(Integer, nullable=False, default=5)
    next_retry_at = Column(DateTime, nullable=True)

    locked_by = Column(String(128), nullable=True)
    locked_at = Column(DateTime, nullable=True)

    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)