"""管理用户种子脚本。
用法: python -m app.scripts.seed_admin [--username admin] [--password xxx]
"""

import argparse

from app.core.database import SessionLocal
from app.core.auth import hash_password
from app.core.config import settings
from app.repository.user_repo import get_user_by_username, create_user


def main():
    parser = argparse.ArgumentParser(description="创建或更新管理员用户")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    username = args.username or settings.initial_admin_username
    password = args.password or settings.initial_admin_password

    if not password:
        print("ERROR: 未提供管理员密码。请在 .env 中设置 INITIAL_ADMIN_PASSWORD 或使用 --password")
        return

    db = SessionLocal()
    try:
        existing = get_user_by_username(db, username)
        if existing:
            existing.password_hash = hash_password(password)
            existing.role = "admin"
            db.commit()
            print(f"管理员用户 '{username}' 已更新。")
        else:
            create_user(db, username, hash_password(password), role="admin")
            print(f"管理员用户 '{username}' 已创建。")
    finally:
        db.close()


if __name__ == "__main__":
    main()
