"""场景知识库 Pydantic Schemas。"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


# ── 场景模板 ──

class ScenarioTemplateCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=128)
    sub_category: str = Field(default="", max_length=128)
    description: str = Field(default="")
    schema_json: str = Field(default="")
    tags: str = Field(default="")
    icon: str = Field(default="")
    priority: int = Field(default=0)


class ScenarioTemplateUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str | None = None
    category: str | None = None
    sub_category: str | None = None
    description: str | None = None
    schema_json: str | None = None
    tags: str | None = None
    icon: str | None = None
    priority: int | None = None
    is_active: bool | None = None


class ScenarioTemplateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    template_uid: str
    name: str
    category: str
    sub_category: str | None = None
    description: str | None = None
    schema_json: str | None = None
    tags: str | None = None
    icon: str | None = None
    priority: int = 0
    is_active: bool = True
    entry_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ScenarioTemplateListResponse(BaseModel):
    templates: List[ScenarioTemplateResponse]
    total: int


# ── 知识条目 ──

class KnowledgeEntryCreate(BaseModel):
    template_uid: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=512)
    content_json: str = Field(..., min_length=1)
    tags: str = Field(default="")
    keywords: str = Field(default="")
    source_type: str = Field(default="manual")
    source_document_uid: str = Field(default="")


class KnowledgeEntryUpdate(BaseModel):
    title: str | None = None
    content_json: str | None = None
    tags: str | None = None
    keywords: str | None = None
    status: str | None = None


class KnowledgeEntryResponse(BaseModel):
    entry_uid: str
    template_uid: str
    title: str
    content_json: str
    plain_text: str
    tags: str | None = None
    keywords: str | None = None
    source_type: str = "manual"
    quality_score: float = 0
    usage_count: int = 0
    helpful_count: int = 0
    status: str = "draft"
    reviewer_id: int | None = None
    reviewed_at: datetime | None = None
    owner_id: int | None = None
    relations: List["KnowledgeRelationResponse"] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeEntryListResponse(BaseModel):
    entries: List[KnowledgeEntryResponse]
    total: int
    page: int
    page_size: int


# ── 知识关联 ──

class KnowledgeRelationCreate(BaseModel):
    entry_uid_from: str = Field(..., min_length=1)
    entry_uid_to: str = Field(..., min_length=1)
    relation_type: str = Field(default="related_to")  # related_to / caused_by / prerequisite_of / similar_to / accompanied_by
    weight: float = Field(default=1.0)


class KnowledgeRelationResponse(BaseModel):
    entry_uid_from: str
    entry_uid_to: str
    relation_type: str
    weight: float


# ── 审核 ──

class ReviewRequest(BaseModel):
    action: str = Field(..., description="approve 或 reject")
    comment: str = Field(default="")


# ── 反馈 ──

class FeedbackRequest(BaseModel):
    helpful: bool


# ── 场景匹配 ──

class ScenarioMatchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0, le=1.0)


class MatchedEntry(BaseModel):
    entry_uid: str
    title: str
    content_json: str
    template_name: str
    template_icon: str = ""
    similarity_score: float
    tags: str = ""


class ScenarioMatchResponse(BaseModel):
    query: str
    entries: List[MatchedEntry]
    match_count: int
    matched_template_uid: str | None = None
    elapsed_ms: float = 0


# ── 通用 ──

class SimpleOKResponse(BaseModel):
    ok: bool
    message: str = ""


class KnowledgeStatsResponse(BaseModel):
    total_templates: int
    total_entries: int
    approved_entries: int
    pending_review_entries: int
    draft_entries: int
    total_match_count: int
    avg_helpful_rate: float = 0
