from sqlalchemy import Column, BigInteger, String, DateTime, Integer, Text, Index, ForeignKey
from app.core.database import Base
from app.core.time_utils import now_shanghai


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    chunk_uid = Column(String(64), unique=True, nullable=False, index=True)

    document_uid = Column(String(64), nullable=False, index=True)
    project_uid = Column(String(64), nullable=False, index=True)
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True, index=True)

    version = Column(Integer, nullable=False, default=1)
    chunk_index = Column(Integer, nullable=False)

    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)

    # 结构信息
    section_title = Column(String(512), nullable=True)
    heading_path = Column(String(1024), nullable=True)
    block_type = Column(String(64), nullable=True)  # paragraph / table / heading / code / mixed
    page_no = Column(Integer, nullable=True)

    # 原文定位
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)

    token_count = Column(Integer, nullable=True)

    embedding_status = Column(String(32), nullable=False, default="pending")
    index_status = Column(String(32), nullable=False, default="pending")

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)


Index(
    "idx_doc_chunk_version_index",
    DocumentChunk.document_uid,
    DocumentChunk.version,
    DocumentChunk.chunk_index,
    unique=True,
)