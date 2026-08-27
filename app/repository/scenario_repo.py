"""场景知识库 Repository — 数据访问层。"""

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.scenario import (
    ScenarioTemplate, KnowledgeEntry, ScenarioMatchLog,
)
from app.models.user import User  # ensure user table in metadata for FK resolution
from app.core.time_utils import now_shanghai


# ── 场景模板 ──

def create_template(db: Session, template_uid: str, owner_id: int, **kwargs) -> ScenarioTemplate:
    tmpl = ScenarioTemplate(
        template_uid=template_uid,
        owner_id=owner_id,
        **kwargs,
    )
    db.add(tmpl)
    db.commit()
    db.refresh(tmpl)
    return tmpl


def get_template_by_uid(db: Session, template_uid: str) -> ScenarioTemplate | None:
    return db.query(ScenarioTemplate).filter(
        ScenarioTemplate.template_uid == template_uid
    ).first()


def list_templates(
    db: Session,
    category: str | None = None,
    active_only: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> list[ScenarioTemplate]:
    q = db.query(ScenarioTemplate)
    if active_only:
        q = q.filter(ScenarioTemplate.is_active == True)
    if category:
        q = q.filter(ScenarioTemplate.category == category)
    return q.order_by(ScenarioTemplate.priority.desc(), ScenarioTemplate.category.asc()).offset(offset).limit(limit).all()


def count_templates(db: Session, category: str | None = None, active_only: bool = True) -> int:
    q = db.query(func.count(ScenarioTemplate.id))
    if active_only:
        q = q.filter(ScenarioTemplate.is_active == True)
    if category:
        q = q.filter(ScenarioTemplate.category == category)
    return q.scalar() or 0


def update_template(db: Session, template_uid: str, **kwargs) -> ScenarioTemplate | None:
    tmpl = get_template_by_uid(db, template_uid)
    if not tmpl:
        return None
    for k, v in kwargs.items():
        if hasattr(tmpl, k):
            setattr(tmpl, k, v)
    tmpl.updated_at = now_shanghai()
    db.commit()
    db.refresh(tmpl)
    return tmpl


# ── 知识条目 ──

def create_entry(db: Session, entry_uid: str, owner_id: int, **kwargs) -> KnowledgeEntry:
    entry = KnowledgeEntry(
        entry_uid=entry_uid,
        owner_id=owner_id,
        **kwargs,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_entry_by_uid(db: Session, entry_uid: str) -> KnowledgeEntry | None:
    return db.query(KnowledgeEntry).filter(
        KnowledgeEntry.entry_uid == entry_uid
    ).first()


def list_entries(
    db: Session,
    template_uid: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> list[KnowledgeEntry]:
    q = db.query(KnowledgeEntry)
    if template_uid:
        q = q.filter(KnowledgeEntry.template_uid == template_uid)
    if status:
        q = q.filter(KnowledgeEntry.status == status)
    if tag:
        q = q.filter(KnowledgeEntry.tags.contains(tag))
    return q.order_by(KnowledgeEntry.updated_at.desc()).offset(offset).limit(limit).all()


def count_entries(
    db: Session,
    template_uid: str | None = None,
    status: str | None = None,
) -> int:
    q = db.query(func.count(KnowledgeEntry.id))
    if template_uid:
        q = q.filter(KnowledgeEntry.template_uid == template_uid)
    if status:
        q = q.filter(KnowledgeEntry.status == status)
    return q.scalar() or 0


def update_entry(db: Session, entry_uid: str, **kwargs) -> KnowledgeEntry | None:
    entry = get_entry_by_uid(db, entry_uid)
    if not entry:
        return None
    for k, v in kwargs.items():
        if hasattr(entry, k):
            setattr(entry, k, v)
    entry.updated_at = now_shanghai()
    db.commit()
    db.refresh(entry)
    return entry


def delete_entry(db: Session, entry_uid: str) -> bool:
    entry = get_entry_by_uid(db, entry_uid)
    if not entry:
        return False
    db.delete(entry)
    db.commit()
    return True


def increment_entry_usage(db: Session, entry_uid: str):
    """匹配命中时 usage_count +1。"""
    entry = get_entry_by_uid(db, entry_uid)
    if entry:
        entry.usage_count += 1
        db.commit()


def mark_entry_helpful(db: Session, entry_uid: str, helpful: bool):
    """用户反馈：helpful_count +1 或 -1。"""
    entry = get_entry_by_uid(db, entry_uid)
    if entry:
        if helpful:
            entry.helpful_count += 1
        else:
            entry.helpful_count = max(0, entry.helpful_count - 1)
        db.commit()


def count_entries_by_status(db: Session) -> dict[str, int]:
    rows = (
        db.query(KnowledgeEntry.status, func.count(KnowledgeEntry.id))
        .group_by(KnowledgeEntry.status)
        .all()
    )
    return {status: count for status, count in rows}


# ── 匹配日志 ──

def create_match_log(
    db: Session,
    query_text: str,
    matched_template_uid: str | None = None,
    matched_entry_uids: str | None = None,
    similarity_score: float | None = None,
    user_id: int | None = None,
    conversation_uid: str | None = None,
) -> ScenarioMatchLog:
    log = ScenarioMatchLog(
        query_text=query_text,
        matched_template_uid=matched_template_uid,
        matched_entry_uids=matched_entry_uids,
        similarity_score=similarity_score,
        user_id=user_id,
        conversation_uid=conversation_uid,
    )
    db.add(log)
    db.commit()
    return log


def get_total_match_count(db: Session) -> int:
    return db.query(func.count(ScenarioMatchLog.id)).scalar() or 0


def get_avg_helpful_rate(db: Session) -> float:
    total = db.query(func.count(ScenarioMatchLog.id)).filter(
        ScenarioMatchLog.was_helpful != None
    ).scalar() or 0
    if total == 0:
        return 0
    helpful = db.query(func.count(ScenarioMatchLog.id)).filter(
        ScenarioMatchLog.was_helpful == True
    ).scalar() or 0
    return helpful / total


# ── 修订历史 ──

def create_revision(
    db: Session,
    entry_uid: str,
    revision_number: int,
    content_json_snapshot: str,
    change_summary: str = "",
    changed_by: int | None = None,
):
    from app.models.scenario import KnowledgeRevision
    rev = KnowledgeRevision(
        entry_uid=entry_uid,
        revision_number=revision_number,
        content_json_snapshot=content_json_snapshot,
        change_summary=change_summary,
        changed_by=changed_by,
    )
    db.add(rev)
    db.commit()
    return rev


def list_revisions(db: Session, entry_uid: str) -> list:
    from app.models.scenario import KnowledgeRevision
    return (
        db.query(KnowledgeRevision)
        .filter(KnowledgeRevision.entry_uid == entry_uid)
        .order_by(KnowledgeRevision.revision_number.desc())
        .all()
    )


def get_revision(db: Session, entry_uid: str, revision_number: int):
    from app.models.scenario import KnowledgeRevision
    return db.query(KnowledgeRevision).filter(
        KnowledgeRevision.entry_uid == entry_uid,
        KnowledgeRevision.revision_number == revision_number,
    ).first()


def get_next_revision_number(db: Session, entry_uid: str) -> int:
    from app.models.scenario import KnowledgeRevision
    max_rev = (
        db.query(func.max(KnowledgeRevision.revision_number))
        .filter(KnowledgeRevision.entry_uid == entry_uid)
        .scalar()
    )
    return (max_rev or 0) + 1
