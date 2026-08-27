from sqlalchemy import Column, BigInteger, String, DateTime, Text, Boolean, Integer, ForeignKey
from app.core.database import Base
from app.core.time_utils import now_shanghai


class Document(Base):
    __tablename__ = "document"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    document_uid = Column(String(64), unique=True, nullable=False, index=True)
    project_uid = Column(String(64), nullable=False, index=True)
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True, index=True)

    filename = Column(String(255), nullable=False)
    file_md5 = Column(String(64), nullable=False, index=True)
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(String(64), nullable=True)

    bucket = Column(String(128), nullable=False)
    object_key = Column(String(512), nullable=False)

    upload_status = Column(String(32), nullable=False, default="success")
    parse_status = Column(String(32), nullable=False, default="pending")
    chunk_status = Column(String(32), nullable=False, default="pending")
    embedding_status = Column(String(32), nullable=False, default="pending")
    index_status = Column(String(32), nullable=False, default="pending")

    current_version = Column(Integer, nullable=False, default=1)
    available_for_search = Column(Boolean, nullable=False, default=False)

    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)