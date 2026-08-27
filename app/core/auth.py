import hashlib
import hmac
import json
import os
import time
from base64 import urlsafe_b64encode, urlsafe_b64decode

from app.core.config import settings


def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return urlsafe_b64decode(data)


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 哈希密码，返回 salt:hash 格式。"""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
    return _b64encode(salt) + ":" + _b64encode(key)


def verify_password(password: str, stored: str) -> bool:
    """验证密码与存储的哈希是否匹配。"""
    try:
        salt_b64, key_b64 = stored.split(":", 1)
        salt = _b64decode(salt_b64)
        expected = _b64decode(key_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_access_token(user_id: int, username: str, role: str) -> str:
    """创建 JWT（HMAC-SHA256 签名）。"""
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64encode(
        json.dumps({
            "sub": user_id,
            "username": username,
            "role": role,
            "iat": now,
            "exp": now + settings.jwt_expire_hours * 3600,
        }).encode()
    )
    signing_input = f"{header}.{payload}"
    sig = hmac.new(
        settings.jwt_secret_key.encode("utf-8"),
        signing_input.encode(),
        "sha256",
    ).digest()
    return f"{signing_input}.{_b64encode(sig)}"


def decode_access_token(token: str) -> dict | None:
    """验证并解码 JWT，返回 payload dict 或 None。"""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            settings.jwt_secret_key.encode("utf-8"),
            signing_input.encode(),
            "sha256",
        ).digest()
        actual_sig = _b64decode(sig_b64)

        if not hmac.compare_digest(actual_sig, expected_sig):
            return None

        payload = json.loads(_b64decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None
