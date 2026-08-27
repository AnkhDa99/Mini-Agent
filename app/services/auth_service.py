import secrets

from sqlalchemy.orm import Session

from app.core.auth import hash_password, verify_password, create_access_token
from app.core.config import settings
from app.repository.user_repo import get_user_by_username, create_user, count_users
from app.repository.document_repo import list_documents_by_owner
from app.schemas.auth import AuthResponse, GuestLoginResponse, UserInfo


class AuthService:
    def register(self, db: Session, username: str, password: str) -> AuthResponse:
        username = username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) < 3:
            raise ValueError("用户名至少 3 个字符")

        existing = get_user_by_username(db, username)
        if existing:
            raise ValueError("用户名已存在")

        # 第一个注册用户自动成为管理员
        role = "admin" if count_users(db) == 0 else "user"

        user = create_user(db, username, hash_password(password), role=role)
        token = create_access_token(user.id, user.username, user.role)
        return AuthResponse(token=token, username=user.username, role=user.role)

    def login(self, db: Session, username: str, password: str) -> AuthResponse:
        user = get_user_by_username(db, username.strip())
        if not user or not user.is_active:
            raise ValueError("用户名或密码错误")
        if not verify_password(password, user.password_hash):
            raise ValueError("用户名或密码错误")

        token = create_access_token(user.id, user.username, user.role)
        return AuthResponse(token=token, username=user.username, role=user.role)

    def guest_login(self, db: Session, guest_user_id: int | None = None) -> GuestLoginResponse:
        # 复用已有游客账户（防止退出重登重置配额）
        if guest_user_id is not None:
            from app.repository.user_repo import get_user_by_id
            existing = get_user_by_id(db, guest_user_id)
            if existing and existing.role == "guest" and existing.is_active:
                token = create_access_token(existing.id, existing.username, existing.role)
                doc_count = len(list_documents_by_owner(db, existing.id))
                return GuestLoginResponse(
                    token=token,
                    user_id=existing.id,
                    username=existing.username,
                    role=existing.role,
                    question_limit=existing.question_limit,
                    question_count=existing.question_count,
                    document_limit=existing.document_limit,
                    document_count=doc_count,
                )

        guest_username = f"guest_{secrets.token_hex(4)}"
        while get_user_by_username(db, guest_username):
            guest_username = f"guest_{secrets.token_hex(4)}"

        random_password = secrets.token_hex(32)
        user = create_user(
            db,
            username=guest_username,
            password_hash=hash_password(random_password),
            role="guest",
            question_limit=settings.guest_question_limit,
            document_limit=settings.guest_document_limit,
        )
        token = create_access_token(user.id, user.username, user.role)

        doc_count = len(list_documents_by_owner(db, user.id))
        return GuestLoginResponse(
            token=token,
            user_id=user.id,
            username=user.username,
            role=user.role,
            question_limit=user.question_limit,
            question_count=user.question_count,
            document_limit=user.document_limit,
            document_count=doc_count,
        )

    def get_user_info(self, user) -> UserInfo:
        return UserInfo(
            id=user.id,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
        )
