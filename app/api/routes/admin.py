import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_admin, get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.time_utils import now_shanghai
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message
from app.models.user import User
from app.repository.document_repo import get_document_by_uid
from app.services.upload_service import UploadService
from pydantic import BaseModel

from app.schemas.auth import UserInfo


class SystemPromptUpdate(BaseModel):
    prompt: str


class RoleUpdate(BaseModel):
    role: str


router = APIRouter(prefix="/admin", tags=["admin"])
upload_service = UploadService()

SYSTEM_PROMPT_FILE = Path(settings.faiss_index_path).parent / "system_prompt.json"


def _load_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        try:
            data = json.loads(SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"))
            return data.get("prompt", "")
        except Exception:
            pass
    return ""


def _save_system_prompt(prompt: str):
    SYSTEM_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_FILE.write_text(
        json.dumps({"prompt": prompt, "updated_at": ""}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@router.get("/system-prompt")
def get_system_prompt(_: User = Depends(get_current_admin)):
    prompt = _load_system_prompt()
    return {"prompt": prompt, "using_default": not prompt}


@router.put("/system-prompt")
def update_system_prompt(
    body: SystemPromptUpdate,
    _: User = Depends(get_current_admin),
):
    prompt = body.prompt.strip()
    _save_system_prompt(prompt)
    settings.system_prompt = prompt
    return {"prompt": prompt, "saved": True}


@router.get("/users", response_model=list[UserInfo])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    users = db.query(User).order_by(User.id.asc()).all()
    return [
        UserInfo(id=u.id, username=u.username, role=u.role, is_active=u.is_active)
        for u in users
    ]


@router.put("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    body: RoleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    new_role = body.role
    if new_role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色必须是 admin 或 user")
    user.role = new_role
    db.commit()
    return {"message": f"用户 {user.username} 角色已更新为 {new_role}"}


@router.get("/documents")
def list_all_documents(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    docs = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .limit(200)
        .all()
    )
    return [
        {
            "document_uid": d.document_uid,
            "filename": d.filename,
            "owner_id": d.owner_id,
            "project_uid": d.project_uid,
            "parse_status": d.parse_status,
            "embedding_status": d.embedding_status,
            "available_for_search": d.available_for_search,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.delete("/documents/{document_uid}")
def admin_delete_document(
    document_uid: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    """管理员可删除任意文档。"""
    try:
        return upload_service.delete_document(db=db, document_uid=document_uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _calc_day_stats(db: Session, day_start: datetime):
    """计算某一天的非管理员流量统计。"""
    day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)
    base = (
        db.query(Message, User.role, Conversation.owner_id, Conversation.id)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .outerjoin(User, Conversation.owner_id == User.id)
        .filter(Message.created_at >= day_start, Message.created_at <= day_end)
        .filter(or_(User.role != "admin", User.role.is_(None)))
        .filter(Message.role == "user")
    )
    rows = base.all()

    total_messages = len(rows)
    owner_ids = set()
    guest_count = 0
    user_count = 0
    hourly_guest = [0] * 24
    hourly_user = [0] * 24

    for msg, role, owner_id, conversation_id in rows:
        owner_ids.add(owner_id if owner_id is not None else f"legacy-{conversation_id}")
        hour = msg.created_at.hour if msg.created_at else 0
        if 0 <= hour < 24:
            if role == "guest":
                guest_count += 1
                hourly_guest[hour] += 1
            else:
                user_count += 1
                hourly_user[hour] += 1

    hourly = [
        {
            "hour": h,
            "label": f"{h:02d}:00",
            "guest_count": hourly_guest[h],
            "user_count": hourly_user[h],
            "total": hourly_guest[h] + hourly_user[h],
        }
        for h in range(24)
    ]

    return {
        "date": day_start.strftime("%Y-%m-%d"),
        "total_messages": total_messages,
        "unique_visitors": len(owner_ids),
        "guest_messages": guest_count,
        "user_messages": user_count,
        "hourly": hourly,
    }


@router.get("/stats")
def traffic_stats(
    db: Session = Depends(get_db),
    date: str | None = None,
    days: int | None = None,
    _: User = Depends(get_current_admin),
):
    """流量统计（排除管理员自身）。不传参=今天，date=指定日期，days=最近N天汇总。"""
    if days:
        days = min(days, 30)  # 最多 30 天
        results = []
        for i in range(days - 1, -1, -1):
            day_start = (now_shanghai() - timedelta(days=i)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            # 只返回摘要，不含 hourly 以减小响应体积
            day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)
            rows = (
                db.query(Message, User.role, Conversation.owner_id, Conversation.id)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .outerjoin(User, Conversation.owner_id == User.id)
                .filter(Message.created_at >= day_start, Message.created_at <= day_end)
                .filter(or_(User.role != "admin", User.role.is_(None)))
                .filter(Message.role == "user")
            ).all()
            owner_ids = set()
            guest_c = 0
            user_c = 0
            for msg, role, owner_id, conversation_id in rows:
                owner_ids.add(owner_id if owner_id is not None else f"legacy-{conversation_id}")
                if role == "guest":
                    guest_c += 1
                else:
                    user_c += 1
            results.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "total_messages": len(rows),
                "unique_visitors": len(owner_ids),
                "guest_messages": guest_c,
                "user_messages": user_c,
            })
        return {"mode": "range", "days": days, "daily": results}

    if date:
        try:
            day_start = datetime.strptime(date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="日期格式错误，应为 YYYY-MM-DD")
    else:
        day_start = now_shanghai().replace(hour=0, minute=0, second=0, microsecond=0)

    return _calc_day_stats(db, day_start)
