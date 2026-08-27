from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str


class GuestLoginRequest(BaseModel):
    guest_user_id: int | None = None


class GuestLoginResponse(BaseModel):
    token: str
    user_id: int
    username: str
    role: str
    question_limit: int
    question_count: int
    document_limit: int
    document_count: int


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
