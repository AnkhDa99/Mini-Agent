from __future__ import annotations

import uuid

from app.core.database import SessionLocal
from app.repository.conversation_repo import (
    create_conversation,
    update_conversation_summary,
)
from app.repository.message_repo import (
    create_message,
    list_recent_messages_by_conversation,
)
from app.repository.chat_cache_repo import (
    set_context_messages,
    touch_recent_conversation,
)
from app.core.config import settings


def build_user_message(i: int, conv_index: int) -> str:
    return (
        f"这是第 {conv_index} 个测试会话中的第 {i} 轮用户消息。"
        f"我正在测试长会话上下文、Redis 最近 {settings.chat_context_limit} 条缓存、"
        f"MySQL 全量持久化以及 summary 拼接能力。"
    )


def build_assistant_message(i: int, conv_index: int) -> str:
    return (
        f"收到，这是第 {conv_index} 个测试会话中的第 {i} 轮助手回复。"
        f"当前对话用于模拟真实长会话场景，验证 Redis 热缓存和 MySQL 回源性能。"
    )


def seed_one_conversation(
    db,
    conv_index: int,
    rounds: int = 100,
    preload_redis: bool = True,
):
    conversation_uid = f"conv_seed_{conv_index}_{uuid.uuid4().hex[:8]}"
    title = f"压测会话 {conv_index}"

    conversation = create_conversation(
        db=db,
        conversation_uid=conversation_uid,
        title=title,
    )

    summary_text = (
        f"这是压测会话 {conv_index} 的历史摘要。"
        f"该会话用于测试长会话下 MySQL 完整历史保存、Redis 最近上下文缓存、"
        f"以及 summary + recent messages 的 Prompt 构造能力。"
    )

    update_conversation_summary(
        db=db,
        conversation_id=conversation.id,
        summary=summary_text,
    )

    for i in range(1, rounds + 1):
        user_msg = build_user_message(i, conv_index)
        assistant_msg = build_assistant_message(i, conv_index)

        create_message(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=user_msg,
        )

        create_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_msg,
        )

    touch_recent_conversation(
        conversation_uid=conversation.conversation_uid,
        title=conversation.title,
    )

    if preload_redis:
        rows = list_recent_messages_by_conversation(
            db=db,
            conversation_id=conversation.id,
            limit=settings.chat_context_limit,
        )

        payload = [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

        set_context_messages(
            conversation_uid=conversation.conversation_uid,
            messages=payload,
        )

    print(
        f"已创建会话: {conversation.conversation_uid}, "
        f"标题: {conversation.title}, "
        f"轮数: {rounds}, "
        f"消息总数: {rounds * 2}, "
        f"Redis预热: {preload_redis}"
    )


def main():
    db = SessionLocal()

    try:
        conversation_count = 10
        rounds_per_conversation = 100

        for conv_index in range(1, conversation_count + 1):
            seed_one_conversation(
                db=db,
                conv_index=conv_index,
                rounds=rounds_per_conversation,
                preload_redis=True,
            )

        print("批量造数完成。")
        print(
            f"共创建 {conversation_count} 个会话，"
            f"每个会话 {rounds_per_conversation * 2} 条消息。"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()