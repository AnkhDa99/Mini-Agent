"""场景知识库核心服务 — 业务逻辑编排。

职责：
- 场景模板 CRUD
- 知识条目 CRUD + 审核 + 修订历史
- Neo4j 知识图谱操作
- 场景匹配（调用 ScenarioMatcher）
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time_utils import now_shanghai
from app.core.neo4j import (
    upsert_entry_node, delete_entry_node,
    create_relation, delete_relation, get_entry_relations,
)
from app.repository import scenario_repo as repo
from app.models.user import User

logger = logging.getLogger(__name__)


def _make_uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ScenarioService:
    """场景知识库服务（无状态方法，db 从外部注入）。"""

    # ═════════════════════════════════════════════════════════════
    # 场景模板
    # ═════════════════════════════════════════════════════════════

    def create_template(
        self, db: Session, user: User, **kwargs,
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以创建场景模板")
        template_uid = _make_uid("tmpl")
        kwargs.pop("template_uid", None)
        tmpl = repo.create_template(
            db, template_uid=template_uid, owner_id=user.id, **kwargs,
        )
        return self._template_to_dict(db, tmpl)

    def get_template(self, db: Session, template_uid: str) -> dict:
        tmpl = repo.get_template_by_uid(db, template_uid)
        if not tmpl:
            raise ValueError(f"场景模板不存在: {template_uid}")
        return self._template_to_dict(db, tmpl)

    def list_templates(
        self, db: Session,
        category: str | None = None,
        active_only: bool = True,
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        offset = (page - 1) * page_size
        templates = repo.list_templates(
            db, category=category, active_only=active_only,
            offset=offset, limit=page_size,
        )
        total = repo.count_templates(db, category=category, active_only=active_only)
        return {
            "templates": [self._template_to_dict(db, t) for t in templates],
            "total": total,
        }

    def update_template(
        self, db: Session, template_uid: str, user: User, **kwargs,
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以更新场景模板")
        tmpl = repo.update_template(db, template_uid, **kwargs)
        if not tmpl:
            raise ValueError(f"场景模板不存在: {template_uid}")
        return self._template_to_dict(db, tmpl)

    def delete_template(
        self, db: Session, template_uid: str, user: User,
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以删除场景模板")
        tmpl = repo.update_template(db, template_uid, is_active=False)
        if not tmpl:
            raise ValueError(f"场景模板不存在: {template_uid}")
        return {"ok": True, "message": f"场景模板 {template_uid} 已停用"}

    def _template_to_dict(self, db: Session, tmpl) -> dict:
        entry_count = repo.count_entries(db, template_uid=tmpl.template_uid)
        return {
            "template_uid": tmpl.template_uid,
            "name": tmpl.name,
            "category": tmpl.category,
            "sub_category": tmpl.sub_category,
            "description": tmpl.description,
            "schema_json": tmpl.schema_json,
            "tags": tmpl.tags,
            "icon": tmpl.icon,
            "priority": tmpl.priority,
            "is_active": tmpl.is_active,
            "entry_count": entry_count,
            "created_at": tmpl.created_at.isoformat() if tmpl.created_at else None,
            "updated_at": tmpl.updated_at.isoformat() if tmpl.updated_at else None,
        }

    # ═════════════════════════════════════════════════════════════
    # 知识条目
    # ═════════════════════════════════════════════════════════════

    def create_entry(
        self, db: Session, user: User, **kwargs,
    ) -> dict:
        entry_uid = _make_uid("entry")

        # 校验模板存在
        template_uid = kwargs.get("template_uid", "")
        tmpl = repo.get_template_by_uid(db, template_uid)
        if not tmpl:
            raise ValueError(f"场景模板不存在: {template_uid}")

        # 从 content_json 生成 plain_text（可读文本，用于 embedding 检索）
        content_json = kwargs.get("content_json", "{}")
        plain_text = self._content_to_plain_text(kwargs.get("title", ""), content_json)

        entry = repo.create_entry(
            db,
            entry_uid=entry_uid,
            owner_id=user.id,
            plain_text=plain_text,
            status="draft",
            **kwargs,
        )

        # 同步到 Neo4j
        upsert_entry_node(
            entry_uid=entry.entry_uid,
            title=entry.title,
            template_uid=entry.template_uid,
        )

        # 加入 FAISS 场景索引（先索引，审核后自动生效）
        self._index_to_faiss(entry)

        return self._entry_to_dict(db, entry)

    def get_entry(self, db: Session, entry_uid: str) -> dict:
        entry = repo.get_entry_by_uid(db, entry_uid)
        if not entry:
            raise ValueError(f"知识条目不存在: {entry_uid}")
        return self._entry_to_dict(db, entry)

    def list_entries(
        self, db: Session,
        template_uid: str | None = None,
        status: str | None = "approved",
        tag: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        offset = (page - 1) * page_size
        entries = repo.list_entries(
            db, template_uid=template_uid, status=status,
            tag=tag, offset=offset, limit=page_size,
        )
        total = repo.count_entries(
            db, template_uid=template_uid, status=status,
        )
        return {
            "entries": [self._entry_to_dict(db, e) for e in entries],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def update_entry(
        self, db: Session, entry_uid: str, user: User, **kwargs,
    ) -> dict:
        entry = repo.get_entry_by_uid(db, entry_uid)
        if not entry:
            raise ValueError(f"知识条目不存在: {entry_uid}")

        # 权限检查：只有 owner 或 admin 可以编辑
        if entry.owner_id != user.id and user.role != "admin":
            raise ValueError("只能编辑自己创建的知识条目")

        # 创建修订历史（在更新前保存快照）
        if "content_json" in kwargs or "title" in kwargs:
            revision_num = repo.get_next_revision_number(db, entry_uid)
            repo.create_revision(
                db,
                entry_uid=entry_uid,
                revision_number=revision_num,
                content_json_snapshot=entry.content_json,
                change_summary=kwargs.get("change_summary", ""),
                changed_by=user.id,
            )

        # 如果 title 或 content_json 变更，更新 plain_text
        if "content_json" in kwargs:
            title = kwargs.get("title", entry.title)
            kwargs["plain_text"] = self._content_to_plain_text(title, kwargs["content_json"])

        entry = repo.update_entry(db, entry_uid, **kwargs)
        if not entry:
            raise ValueError(f"更新失败: {entry_uid}")

        # 同步 Neo4j
        upsert_entry_node(
            entry_uid=entry.entry_uid,
            title=entry.title,
            template_uid=entry.template_uid,
        )

        return self._entry_to_dict(db, entry)

    def delete_entry(
        self, db: Session, entry_uid: str, user: User,
    ) -> dict:
        entry = repo.get_entry_by_uid(db, entry_uid)
        if not entry:
            raise ValueError(f"知识条目不存在: {entry_uid}")
        if entry.owner_id != user.id and user.role != "admin":
            raise ValueError("只能删除自己创建的知识条目")

        delete_entry_node(entry_uid)
        repo.delete_entry(db, entry_uid)
        return {"ok": True, "message": f"知识条目 {entry_uid} 已删除"}

    # ── 审核 ──

    def review_entry(
        self, db: Session, entry_uid: str, user: User,
        action: str, comment: str = "",
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以审核")

        if action == "approve":
            new_status = "approved"
            msg = "审核通过"
        elif action == "reject":
            new_status = "draft"
            msg = f"审核不通过: {comment}" if comment else "审核不通过"
        else:
            raise ValueError(f"无效审核动作: {action}，应为 approve 或 reject")

        entry = repo.update_entry(
            db, entry_uid,
            status=new_status,
            reviewer_id=user.id,
            reviewed_at=now_shanghai(),
        )
        if not entry:
            raise ValueError(f"知识条目不存在: {entry_uid}")

        # 审核通过：确保条目在 FAISS 索引中
        if action == "approve":
            self._index_to_faiss(entry)

        return {"ok": True, "message": msg, "status": new_status}

    # ── 反馈 ──

    def submit_feedback(
        self, db: Session, entry_uid: str, helpful: bool,
    ) -> dict:
        repo.mark_entry_helpful(db, entry_uid, helpful)
        return {"ok": True, "message": "感谢反馈"}

    # ── 修订历史 ──

    def list_revisions(self, db: Session, entry_uid: str) -> list[dict]:
        _ = repo.get_entry_by_uid(db, entry_uid)  # 校验条目存在
        revs = repo.list_revisions(db, entry_uid)
        return [
            {
                "revision_number": r.revision_number,
                "change_summary": r.change_summary,
                "changed_by": r.changed_by,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in revs
        ]

    def rollback_entry(
        self, db: Session, entry_uid: str, revision_number: int, user: User,
    ) -> dict:
        rev = repo.get_revision(db, entry_uid, revision_number)
        if not rev:
            raise ValueError(f"修订版本不存在: {entry_uid} rev={revision_number}")

        # 先创建当前版本的新修订记录
        entry = repo.get_entry_by_uid(db, entry_uid)
        next_rev = repo.get_next_revision_number(db, entry_uid)
        repo.create_revision(
            db, entry_uid=entry_uid,
            revision_number=next_rev,
            content_json_snapshot=entry.content_json,
            change_summary=f"回滚到版本 {revision_number}",
            changed_by=user.id,
        )

        # 恢复快照
        title = entry.title  # title 从快照中没法直接恢复，保持不变
        new_plain = self._content_to_plain_text(title, rev.content_json_snapshot or "{}")
        repo.update_entry(
            db, entry_uid,
            content_json=rev.content_json_snapshot,
            plain_text=new_plain,
            status="draft",
        )
        return {"ok": True, "message": f"已回滚到版本 {revision_number}"}

    # ═════════════════════════════════════════════════════════════
    # 知识图谱（Neo4j）
    # ═════════════════════════════════════════════════════════════

    def create_entry_relation(
        self, db: Session, user: User, **kwargs,
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以管理知识图谱关联")

        entry_uid_from = kwargs.get("entry_uid_from", "")
        entry_uid_to = kwargs.get("entry_uid_to", "")

        # 校验两个条目都存在
        for uid in (entry_uid_from, entry_uid_to):
            if not repo.get_entry_by_uid(db, uid):
                raise ValueError(f"知识条目不存在: {uid}")

        create_relation(
            entry_uid_from=entry_uid_from,
            entry_uid_to=entry_uid_to,
            relation_type=kwargs.get("relation_type", "related_to"),
            weight=kwargs.get("weight", 1.0),
        )
        return {"ok": True, "message": "关联已创建"}

    def delete_entry_relation(
        self, db: Session, entry_uid_from: str,
        entry_uid_to: str, relation_type: str | None, user: User,
    ) -> dict:
        if user.role != "admin":
            raise ValueError("只有管理员可以管理知识图谱关联")
        delete_relation(entry_uid_from, entry_uid_to, relation_type)
        return {"ok": True, "message": "关联已删除"}

    def get_entry_relations(
        self, db: Session, entry_uid: str, depth: int = 1,
    ) -> list[dict]:
        _ = repo.get_entry_by_uid(db, entry_uid)
        return get_entry_relations(entry_uid, depth=depth)

    # ═════════════════════════════════════════════════════════════
    # 统计
    # ═════════════════════════════════════════════════════════════

    def get_stats(self, db: Session) -> dict:
        status_counts = repo.count_entries_by_status(db)
        return {
            "total_templates": repo.count_templates(db, active_only=False),
            "total_entries": sum(status_counts.values()),
            "approved_entries": status_counts.get("approved", 0),
            "pending_review_entries": status_counts.get("pending_review", 0),
            "draft_entries": status_counts.get("draft", 0),
            "total_match_count": repo.get_total_match_count(db),
            "avg_helpful_rate": round(repo.get_avg_helpful_rate(db), 2),
        }

    # ═════════════════════════════════════════════════════════════
    # 工具方法
    # ═════════════════════════════════════════════════════════════

    def _entry_to_dict(self, db: Session, entry) -> dict:
        relations = get_entry_relations(entry.entry_uid, depth=1)
        return {
            "entry_uid": entry.entry_uid,
            "template_uid": entry.template_uid,
            "title": entry.title,
            "content_json": entry.content_json,
            "plain_text": entry.plain_text,
            "tags": entry.tags,
            "keywords": entry.keywords,
            "source_type": entry.source_type,
            "quality_score": entry.quality_score,
            "usage_count": entry.usage_count,
            "helpful_count": entry.helpful_count,
            "status": entry.status,
            "reviewer_id": entry.reviewer_id,
            "reviewed_at": entry.reviewed_at.isoformat() if entry.reviewed_at else None,
            "owner_id": entry.owner_id,
            "relations": relations,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
        }

    def _index_to_faiss(self, entry) -> None:
        """将条目加入 FAISS 场景索引（失败不影响主流程）。"""
        try:
            from app.services.scenario_matcher import get_scenario_matcher
            matcher = get_scenario_matcher()
            matcher.index_entry(entry.entry_uid, entry.plain_text)
            matcher._save()
            logger.debug("FAISS indexed entry: %s", entry.entry_uid)
        except Exception:
            logger.warning("FAISS index_entry failed for %s (embedding API may not be ready)", entry.entry_uid)

    def rebuild_faiss_index(self) -> dict:
        """全量重建 FAISS 场景索引。"""
        try:
            from app.services.scenario_matcher import get_scenario_matcher
            matcher = get_scenario_matcher()
            matcher.rebuild_index()
            return {"ok": True, "message": f"索引重建完成，共 {matcher.index.ntotal} 条"}
        except Exception as e:
            logger.exception("Rebuild FAISS index failed")
            return {"ok": False, "message": str(e)}

    @staticmethod
    def _content_to_plain_text(title: str, content_json: str) -> str:
        """将结构化 JSON 转换为可检索的纯文本。"""
        try:
            data = json.loads(content_json)
        except json.JSONDecodeError:
            return f"{title}\n{content_json}"

        parts = [title]
        for key, value in data.items():
            if isinstance(value, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in value)}")
            elif isinstance(value, str):
                parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {'; '.join(f'{k}={v}' for k, v in value.items())}")
        return "\n".join(parts)
