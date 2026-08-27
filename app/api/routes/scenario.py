"""场景知识库 API 路由。"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_current_admin
from app.core.database import get_db
from app.models.user import User
from app.schemas.scenario import (
    ScenarioTemplateCreate, ScenarioTemplateUpdate, ScenarioTemplateListResponse,
    KnowledgeEntryCreate, KnowledgeEntryUpdate,
    KnowledgeEntryListResponse,
    KnowledgeRelationCreate, KnowledgeRelationResponse,
    ReviewRequest, FeedbackRequest,
    ScenarioMatchRequest, ScenarioMatchResponse,
    SimpleOKResponse, KnowledgeStatsResponse,
)
from app.services.scenario_service import ScenarioService

router = APIRouter(tags=["scenario_kb"])
service = ScenarioService()


# ── 场景模板 ──

@router.get("/api/scenarios")
def list_templates(
    category: str | None = Query(default=None),
    active_only: bool = Query(default=True),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return service.list_templates(
        db, category=category, active_only=active_only,
        page=page, page_size=page_size,
    )


@router.get("/api/scenarios/{template_uid}")
def get_template(
    template_uid: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return service.get_template(db, template_uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/api/scenarios")
def create_template(
    body: ScenarioTemplateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.create_template(
            db, user=current_user,
            name=body.name,
            category=body.category,
            sub_category=body.sub_category or None,
            description=body.description or None,
            schema_json=body.schema_json or None,
            tags=body.tags or None,
            icon=body.icon or None,
            priority=body.priority,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/api/scenarios/{template_uid}")
def update_template(
    template_uid: str,
    body: ScenarioTemplateUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return service.update_template(db, template_uid, user=current_user, **updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/scenarios/{template_uid}")
def delete_template(
    template_uid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.delete_template(db, template_uid, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── 知识条目 ──

@router.get("/api/knowledge")
def list_entries(
    template_uid: str | None = Query(default=None),
    status: str | None = Query(default="approved"),
    tag: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return service.list_entries(
        db, template_uid=template_uid, status=status,
        tag=tag, page=page, page_size=page_size,
    )


@router.get("/api/knowledge/{entry_uid}")
def get_entry(
    entry_uid: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return service.get_entry(db, entry_uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/api/knowledge")
def create_entry(
    body: KnowledgeEntryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.create_entry(
            db, user=current_user,
            template_uid=body.template_uid,
            title=body.title,
            content_json=body.content_json,
            tags=body.tags or None,
            keywords=body.keywords or None,
            source_type=body.source_type,
            source_document_uid=body.source_document_uid or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/api/knowledge/{entry_uid}")
def update_entry(
    entry_uid: str,
    body: KnowledgeEntryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return service.update_entry(db, entry_uid, user=current_user, **updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/knowledge/{entry_uid}")
def delete_entry(
    entry_uid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.delete_entry(db, entry_uid, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/knowledge/{entry_uid}/review")
def review_entry(
    entry_uid: str,
    body: ReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.review_entry(
            db, entry_uid, user=current_user,
            action=body.action, comment=body.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/knowledge/{entry_uid}/feedback")
def submit_feedback(
    entry_uid: str,
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return service.submit_feedback(db, entry_uid, body.helpful)


@router.get("/api/knowledge/{entry_uid}/revisions")
def list_revisions(
    entry_uid: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return service.list_revisions(db, entry_uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/api/knowledge/{entry_uid}/revisions/{revision_number}/rollback")
def rollback_entry(
    entry_uid: str,
    revision_number: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.rollback_entry(db, entry_uid, revision_number, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── 知识图谱关联 ──

@router.post("/api/knowledge/relations")
def create_relation(
    body: KnowledgeRelationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.create_entry_relation(
            db, user=current_user,
            entry_uid_from=body.entry_uid_from,
            entry_uid_to=body.entry_uid_to,
            relation_type=body.relation_type,
            weight=body.weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/api/knowledge/relations")
def delete_relation(
    entry_uid_from: str = Query(...),
    entry_uid_to: str = Query(...),
    relation_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.delete_entry_relation(
            db, entry_uid_from, entry_uid_to,
            relation_type=relation_type, user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/knowledge/{entry_uid}/relations", response_model=list[KnowledgeRelationResponse])
def get_entry_relations(
    entry_uid: str,
    depth: int = Query(default=1, ge=1, le=3),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return service.get_entry_relations(db, entry_uid, depth=depth)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── 场景匹配（检索接口） ──

@router.post("/api/scenarios/match", response_model=ScenarioMatchResponse)
def match_scenario(
    body: ScenarioMatchRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    try:
        from app.services.scenario_matcher import ScenarioMatcher
        matcher = ScenarioMatcher()
        return matcher.match(
            db=db,
            query=body.query,
            user=current_user,
            top_k=body.top_k,
            threshold=body.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── 管理统计 ──

@router.get("/admin/knowledge/stats", response_model=KnowledgeStatsResponse)
def knowledge_stats(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    return service.get_stats(db)


@router.post("/admin/knowledge/rebuild-index")
def rebuild_knowledge_index(
    _: User = Depends(get_current_admin),
):
    """手动重建 FAISS 场景索引（从 MySQL 读取所有 approved 条目）。"""
    return service.rebuild_faiss_index()


@router.get("/admin/knowledge/pending-review")
def list_pending_review(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    return service.list_entries(
        db, status="pending_review", page=page, page_size=page_size,
    )
