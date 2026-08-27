from sqlalchemy import (
    Column,
    BigInteger,
    String,
    DateTime,
    Integer,
    Text,
    Index,
    ForeignKey,
)
from app.core.database import Base
from app.core.time_utils import now_shanghai


UPLOAD_STATUS_UPLOADING = "uploading"
UPLOAD_STATUS_MERGING = "merging"
UPLOAD_STATUS_COMPLETED = "completed"
UPLOAD_STATUS_FAILED = "failed"
UPLOAD_STATUS_EXPIRED = "expired"


class UploadSession(Base):
    __tablename__ = "upload_session"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # 对外暴露的上传会话 ID
    upload_uid = Column(String(64), unique=True, nullable=False, index=True)

    # 第一版没有用户系统，先用 project_uid 做隔离
    project_uid = Column(String(64), nullable=False, index=True, default="default_project")
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True, index=True)

    # 文件基础信息
    filename = Column(String(255), nullable=False)
    file_md5 = Column(String(64), nullable=False, index=True)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(128), nullable=True)

    # 分片信息
    chunk_size = Column(BigInteger, nullable=False)
    total_parts = Column(Integer, nullable=False)
    uploaded_parts = Column(Integer, nullable=False, default=0)

    # 状态机
    status = Column(String(32), nullable=False, default=UPLOAD_STATUS_UPLOADING)

    # complete_upload 后会生成 document_uid
    document_uid = Column(String(64), nullable=True)

    # 失败原因
    last_error = Column(Text, nullable=True)

    # 上传会话过期时间，后续清理临时分片用
    expires_at = Column(DateTime, nullable=False)

    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)


Index(
    "idx_upload_project_file",
    UploadSession.project_uid,
    UploadSession.file_md5,
    UploadSession.filename,
)

# upload_uid：后续每个分片都带这个 ID
# file_md5：完整文件 MD5，用来做秒传/重复上传判断
# status：后续 complete_upload 做 CAS 抢占
# expires_at：上传中断后清理临时资源