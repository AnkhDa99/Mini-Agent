from sqlalchemy import Column, BigInteger, String, DateTime, Boolean, Integer
from app.core.database import Base
from app.core.time_utils import now_shanghai


class User(Base):
    __tablename__ = "user"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(16), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    question_count = Column(Integer, nullable=False, default=0)
    question_limit = Column(Integer, nullable=False, default=0)  # 0 = unlimited
    document_limit = Column(Integer, nullable=False, default=0)  # 0 = unlimited
    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)
