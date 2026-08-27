from sqlalchemy import (
    Column,
    BigInteger,
    String,
    DateTime,
    Integer,
    Index,
)
from app.core.database import Base
from app.core.time_utils import now_shanghai


UPLOAD_PART_STATUS_SUCCESS = "success"
UPLOAD_PART_STATUS_FAILED = "failed"


class UploadPart(Base):
    __tablename__ = "upload_part"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    upload_uid = Column(String(64), nullable=False, index=True)
    part_number = Column(Integer, nullable=False)

    part_size = Column(BigInteger, nullable=False)
    part_md5 = Column(String(64), nullable=False)

    # MinIO 临时分片路径，例如 tmp/upload_xxx/part-0
    object_key = Column(String(512), nullable=False)

    status = Column(String(32), nullable=False, default=UPLOAD_PART_STATUS_SUCCESS)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)


Index(
    "idx_upload_part_unique",
    UploadPart.upload_uid,
    UploadPart.part_number,
    unique=True,
)