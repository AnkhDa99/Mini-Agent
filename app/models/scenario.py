"""场景知识库数据模型 — 智能运维排障知识库。

MySQL 存储：场景模板 + 知识条目 + 匹配日志 + 修订历史
Neo4j 存储：知识关联图谱
"""

from sqlalchemy import (
    Column, BigInteger, String, DateTime, Text, Boolean,
    Integer, Float, ForeignKey,
)

from app.core.database import Base
from app.core.time_utils import now_shanghai


class ScenarioTemplate(Base):
    """场景模板 — 定义排障知识的一级/二级分类结构。"""

    __tablename__ = "scenario_template"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    template_uid = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(128), nullable=False, index=True)
    sub_category = Column(String(128), nullable=True)
    description = Column(Text, nullable=True)
    schema_json = Column(Text, nullable=True)  # 知识条目 JSON Schema
    tags = Column(String(512), nullable=True)
    icon = Column(String(64), nullable=True)
    priority = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)


class KnowledgeEntry(Base):
    """知识条目 — 每个场景下的具体排障知识卡片。"""

    __tablename__ = "knowledge_entry"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    entry_uid = Column(String(64), unique=True, nullable=False, index=True)
    template_uid = Column(String(64), nullable=False, index=True)
    title = Column(String(512), nullable=False)
    content_json = Column(Text, nullable=False)
    plain_text = Column(Text, nullable=False)
    tags = Column(String(512), nullable=True)
    keywords = Column(String(512), nullable=True)
    source_type = Column(String(32), nullable=False, default="manual")
    source_document_uid = Column(String(64), nullable=True)
    quality_score = Column(Float, nullable=False, default=0)
    usage_count = Column(Integer, nullable=False, default=0)
    helpful_count = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="draft", index=True)
    reviewer_id = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    owner_id = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_shanghai)
    updated_at = Column(DateTime, nullable=False, default=now_shanghai, onupdate=now_shanghai)


class ScenarioMatchLog(Base):
    """场景匹配记录 — 用于优化匹配算法和效果分析。"""

    __tablename__ = "scenario_match_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    matched_template_uid = Column(String(64), nullable=True)
    matched_entry_uids = Column(Text, nullable=True)  # JSON 数组
    similarity_score = Column(Float, nullable=True)
    was_helpful = Column(Boolean, nullable=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    conversation_uid = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_shanghai)


class KnowledgeRevision(Base):
    """知识修订历史 — 支持版本回滚。"""

    __tablename__ = "knowledge_revision"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    entry_uid = Column(String(64), nullable=False, index=True)
    revision_number = Column(Integer, nullable=False)
    content_json_snapshot = Column(Text, nullable=True)
    change_summary = Column(String(512), nullable=True)
    changed_by = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_shanghai)
