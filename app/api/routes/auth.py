from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.repository.document_repo import list_documents_by_owner
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponse, GuestLoginRequest, GuestLoginResponse, UserInfo
from app.services.auth_service import AuthService

router = APIRouter(tags=["auth"])
service = AuthService()


@router.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if not settings.registration_enabled:
        raise HTTPException(status_code=403, detail="注册功能暂时关闭，请使用游客登录或联系管理员获取正式账号")
    try:
        return service.register(db, req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    try:
        return service.login(db, req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/auth/guest-login", response_model=GuestLoginResponse)
def guest_login(req: GuestLoginRequest | None = None, db: Session = Depends(get_db)):
    try:
        return service.guest_login(db, guest_user_id=req.guest_user_id if req else None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"游客登录失败：{exc}") from exc


@router.get("/auth/quota")
def quota(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "guest":
        return {
            "role": current_user.role,
            "question_limit": 0,
            "question_count": 0,
            "document_limit": 0,
            "document_count": 0,
        }
    doc_count = len(list_documents_by_owner(db, current_user.id))
    return {
        "role": "guest",
        "question_limit": current_user.question_limit,
        "question_count": current_user.question_count,
        "document_limit": current_user.document_limit,
        "document_count": doc_count,
    }


@router.get("/auth/me", response_model=UserInfo)
def me(current_user: User = Depends(get_current_user)):
    return service.get_user_info(current_user)
