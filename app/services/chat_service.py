from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.openai_client import OpenAIChatClient
from app.llm.model_registry import get_model_registry
from app.prompts.system_prompts import DEFAULT_SYSTEM_PROMPT, HYDE_PROMPT, MULTI_QUERY_PROMPT


def _get_system_prompt() -> str:
    """从持久化文件或配置中加载 system prompt，空则回退到代码默认值。"""
    prompt = settings.system_prompt
    if prompt:
        return prompt
    prompt_file = Path(settings.faiss_index_path).parent / "system_prompt.json"
    if prompt_file.exists():
        try:
            data = json.loads(prompt_file.read_text(encoding="utf-8"))
            if data.get("prompt"):
                return data["prompt"]
        except Exception:
            pass
    return DEFAULT_SYSTEM_PROMPT

from app.services.cache_invalidation_service import delete_cache_with_retry_task
from app.services.hybrid_search import HybridSearchService
from app.services.search_router import SearchRouter, SearchContext, get_search_router
from app.services.query_classifier import QueryComplexity
from app.services.doc_summary_index import get_doc_summary_index
from app.agent.agent_loop import (
    run_agent_loop, AgentState,
    _generate_final_answer, _generate_final_answer_stream,
)
from app.agent.classifier import ClassificationResult, QueryClass
from app.schemas.chat import (
    ChatResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationSummary,
    MessageItem,
    SourceReference,
)
from app.repository.chat_cache_repo import (
    append_context_message,
    delete_conversation_cache,
    get_cached_context_messages,
    set_context_messages,
    touch_recent_conversation,
)
from app.repository.message_repo import (
    create_message,
    get_message_by_id,
    list_messages_by_conversation,
    list_recent_messages_by_conversation,
    count_messages_by_conversation,
    delete_messages_by_ids,
    get_last_assistant_message,
    get_last_user_message,
    update_message_content,
    delete_messages_after_id,
)
from app.repository.conversation_repo import (
    create_conversation,
    get_by_uid,
    list_recent_conversations,
    touch_conversation,
    update_conversation_title,
    update_conversation_summary,
    delete_conversation_and_messages,
)

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self) -> None:
        self.llm_client = OpenAIChatClient()  # 向后兼容，旧版单一模型
        self._hybrid_search: HybridSearchService | None = None
        self._search_router: SearchRouter | None = None

    @property
    def generator_client(self):
        """多Agent架构：回答生成用 pro 模型。"""
        return get_model_registry().get_client("generator")

    @property
    def hybrid_search(self) -> HybridSearchService:
        if self._hybrid_search is None:
            self._hybrid_search = HybridSearchService()
        return self._hybrid_search

    @property
    def search_router(self) -> SearchRouter:
        if self._search_router is None:
            self._search_router = get_search_router()
        return self._search_router

    def _generate_conversation_uid(self) -> str:
        return f"conv_{uuid.uuid4().hex[:12]}"

    def _safe_append_context_message(
            self,
            conversation_uid: str,
            role: str,
            content: str,
            created_at=None,
    ):
        try:
            append_context_message(
                conversation_uid=conversation_uid,
                role=role,
                content=content,
                created_at=created_at,
            )
        except Exception as exc:
            print(f"【Redis】追加上下文缓存失败，不影响主流程: {exc}")

    def _safe_touch_recent_conversation(
            self,
            conversation_uid: str,
            title: str | None,
    ):
        try:
            touch_recent_conversation(
                conversation_uid=conversation_uid,
                title=title,
            )
        except Exception as exc:
            print(f"【Redis】更新最近会话缓存失败，不影响主流程: {exc}")

    def _safe_set_context_messages(
            self,
            conversation_uid: str,
            messages: list[dict],
    ):
        try:
            set_context_messages(conversation_uid, messages)
        except Exception as exc:
            print(f"【Redis】回填上下文缓存失败，不影响主流程: {exc}")

    def chat(
            self,
            db: Session,
            message: str,
            conversation_id: str | None = None,
            user=None,
            web_search: bool = False,
    ) -> ChatResponse:
        user_message = message.strip()
        if not user_message:
            raise ValueError("请输入问题")

        conversation = None
        is_new_conversation = False
        conv_owner_id = user.id if user else None
        # 搜索范围：管理员跨用户检索，普通用户仅检索自己的文档
        search_owner_id = None if (user and getattr(user, 'role', None) == 'admin') else conv_owner_id

        if conversation_id:
            conversation = get_by_uid(db, conversation_id, owner_id=conv_owner_id)

        if conversation is None:
            is_new_conversation = True
            conversation_id = self._generate_conversation_uid()
            conversation = create_conversation(
                db=db,
                conversation_uid=conversation_id,
                title=None,
                owner_id=conv_owner_id,
            )
        else:
            conversation_id = conversation.conversation_uid

        # 1. 先写 MySQL
        user_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )

        # 2. 再增量写 Redis
        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="user",
                content=user_message,
                created_at=user_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】追加 user 消息缓存失败，不影响主流程: {exc}")

        if is_new_conversation and not conversation.title:
            conversation = update_conversation_title(
                db,
                conversation.id,
                self._build_title(user_message)
            ) or conversation

        # 3. 视情况刷新 summary
        self._maybe_refresh_summary(db, conversation)

        # 4. 只取 summary + 最近20条
        recent_messages = self._load_recent_context(db, conversation)

        # 5. Agent 循环（分类 → 检索 → Think-Act-Observe → 回答）
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        agent_state = self._run_agent_loop(
            db, user_message, recent_messages,
            owner_id=search_owner_id,
            web_search=web_search and allow_web_search,
            allow_web_search=allow_web_search,
        )

        answer = agent_state.final_answer
        search_results = agent_state.all_results

        logger.info(
            "chat start | conversation_id=%s | recent_message_count=%s | search_results=%s | route=%s",
            conversation_id,
            len(recent_messages),
            len(search_results),
            agent_state.classification.query_class.value,
        )

        # 6. 先写 MySQL。若 LLM 判断检索结果无关，抑制来源展示
        sources = [] if agent_state.suppress_sources else self._build_sources(search_results)
        assistant_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources_json=self._sources_to_json(sources),
        )

        # 7. 再增量写 Redis
        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=answer,
                created_at=assistant_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】追加 assistant 消息缓存失败，不影响主流程: {exc}")

        touch_conversation(db, conversation.id)
        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid,
            title=conversation.title,
        )

        logger.info("chat end | conversation_id=%s", conversation_id)

        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            messages_saved=2,
            sources=sources,
        )

    def _build_title(self, message: str) -> str:
        title = message.strip().replace("\n", " ")[:16]
        return title or "新会话"

    def get_recent_conversations(
            self,
            db: Session,
            limit: int = 50,
            user=None,
    ) -> ConversationListResponse:
        owner_id = user.id if user else None
        rows = list_recent_conversations(db, limit=limit, owner_id=owner_id)

        return ConversationListResponse(
            conversations=[
                ConversationSummary(
                    conversation_id=row.conversation_uid,
                    title=row.title or "新会话",
                    updated_at=row.updated_at,
                )
                for row in rows
            ]
        )

    def get_conversation_messages(
            self,
            db: Session,
            conversation_id: str,
            user=None,
    ) -> ConversationMessagesResponse:
        owner_id = user.id if user else None
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        rows = list_messages_by_conversation(db, conversation.id)

        recent_messages = [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows[-settings.chat_context_limit:]
            if row.role in {"user", "assistant", "system"}
        ]
        try:
            self._safe_set_context_messages(conversation.conversation_uid, recent_messages)
        except Exception as exc:
            print(f"【Redis】会话消息回填缓存失败，但不影响主流程: {exc}")

        return ConversationMessagesResponse(
            conversation_id=conversation.conversation_uid,
            title=conversation.title or "新会话",
            messages=[
                MessageItem(
                    id=row.id,
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    sources=self._parse_sources_json(row.sources_json),
                    metadata=self._parse_metadata_json(row.metadata_json),
                )
                for row in rows
            ],
        )

    def stream_chat(
            self,
            db: Session,
            message: str,
            conversation_id: str | None = None,
            user=None,
            web_search: bool = False,
            model_config: dict | None = None,
    ):
        user_message = message.strip()
        if not user_message:
            raise ValueError("请输入问题")

        conversation = None
        is_new_conversation = False
        conv_owner_id = user.id if user else None
        search_owner_id = None if (user and getattr(user, 'role', None) == 'admin') else conv_owner_id

        if conversation_id:
            conversation = get_by_uid(db, conversation_id, owner_id=conv_owner_id)

        if conversation is None:
            is_new_conversation = True
            conversation_id = self._generate_conversation_uid()
            conversation = create_conversation(
                db=db,
                conversation_uid=conversation_id,
                title=None,
                owner_id=conv_owner_id,
            )
        else:
            conversation_id = conversation.conversation_uid

        # 1. 先写 MySQL
        user_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )

        # 2. 再增量写 Redis
        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="user",
                content=user_message,
                created_at=user_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】追加 user 消息缓存失败，不影响主流程: {exc}")

        if is_new_conversation and not conversation.title:
            conversation = update_conversation_title(
                db,
                conversation.id,
                self._build_title(user_message)
            ) or conversation

        # 3. 视情况刷新 summary
        self._maybe_refresh_summary(db, conversation)

        # 4. summary + 最近20条
        recent_messages = self._load_recent_context(db, conversation)

        # 5. Agent 循环（流式：实时推送思考过程 + 最终回答）
        yield {"type": "meta", "conversation_id": conversation_id}

        state_container = [None]
        search_results = []
        full_answer = ""
        sources_data = []
        allow_web_search = not (user and getattr(user, "role", None) == "guest")

        for event in self._run_agent_loop_stream(db, user_message, recent_messages,
                                                  state_container=state_container, owner_id=search_owner_id,
                                                  web_search=web_search and allow_web_search,
                                                  allow_web_search=allow_web_search,
                                                  model_config=model_config):
            if isinstance(event, str):
                full_answer += event
                yield event
            elif isinstance(event, dict):
                etype = event.get("type", "")
                if etype == "sources":
                    sources_data = event.get("sources", [])
                    search_results = [{"chunk_uid": s.get("chunk_uid", ""),
                                       "document_uid": s.get("document_uid", ""),
                                       "filename": s.get("filename", ""),
                                       "content": s.get("content_preview", ""),
                                       "section_title": s.get("section_title", ""),
                                       "page_no": s.get("page_no", 0),
                                       "rrf_score": s.get("score", 0)}
                                      for s in sources_data]
                    yield event
                elif etype == "scenario_matches":
                    yield event
                elif etype == "agent_status":
                    yield event
                elif etype == "usage_metadata":
                    yield event
                elif etype == "status":
                    yield event
                elif etype == "error":
                    yield event
                    return

        agent_state = state_container[0]
        if agent_state is None:
            # 兜底
            agent_state = AgentState(
                user_query=user_message,
                classification=ClassificationResult(
                    query_class=QueryClass.SHALLOW, confidence=0.3,
                    reason="stream-fallback", max_rounds=1, max_tool_calls_per_round=1,
                    allowed_tool_categories=["search"], degradation_max_level=1,
                ),
                round=1, max_rounds=1,
                search_results=search_results, all_results=search_results,
                final_answer=full_answer,
            )

        # 6. 先写 MySQL。若 LLM 判断检索结果无关，抑制来源展示
        sources = [] if agent_state.suppress_sources else self._build_sources(search_results)
        metadata = self._build_usage_metadata(agent_state, sources)
        assistant_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=full_answer,
            sources_json=self._sources_to_json(sources),
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )

        # 6. 再增量写 Redis
        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=full_answer,
                created_at=assistant_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】追加 assistant 消息缓存失败，不影响主流程: {exc}")

        touch_conversation(db, conversation.id)
        try:
            self._safe_touch_recent_conversation(
                conversation_uid=conversation.conversation_uid,
                title=conversation.title,
            )
        except Exception as exc:
            print(f"【Redis】更新最近会话缓存失败，不影响主流程: {exc}")

    def _build_summary_prompt_messages(
            self,
            old_summary: str | None,
            messages: list[dict],
    ) -> list[dict]:
        content_lines = []
        if old_summary:
            content_lines.append(f"已有摘要：\n{old_summary}")
        content_lines.append("请把下面对话压缩成简洁摘要，保留用户目标、约束、上下文结论。")
        for msg in messages:
            content_lines.append(f"{msg['role']}: {msg['content']}")
        joined = "\n".join(content_lines)

        return [
            {
                "role": "system",
                "content": "你是会话摘要助手。输出简洁中文摘要，不要编造。",
            },
            {
                "role": "user",
                "content": joined,
            },
        ]

    # 摘要触发阈值：上下文总字符数超过此值 → 压缩旧消息为摘要
    _MAX_CONTEXT_CHARS = 50000   # ~12K tokens
    _KEEP_RECENT_CHARS = 30000   # ~7.5K tokens 保留为最近消息

    def _maybe_refresh_summary(self, db: Session, conversation) -> None:
        total = count_messages_by_conversation(db, conversation.id)
        if total <= settings.summary_trigger_message_count:
            return

        all_rows = list_messages_by_conversation(db, conversation.id)
        total_chars = sum(len(row.content or "") for row in all_rows)

        # 上下文总量未超阈值，无需压缩
        if total_chars <= self._MAX_CONTEXT_CHARS:
            return

        # 从最新向最旧累计字符数，保留最近 ~30K chars，其余压缩为摘要
        kept_chars = 0
        split_idx = len(all_rows)
        for i, row in enumerate(reversed(all_rows)):
            kept_chars += len(row.content or "")
            if kept_chars > self._KEEP_RECENT_CHARS:
                split_idx = len(all_rows) - i - 1
                break

        if split_idx <= 0:
            return

        old_rows = all_rows[:split_idx]
        old_messages = [
            {"role": row.role, "content": row.content}
            for row in old_rows
            if row.role in {"user", "assistant", "system"}
        ]
        if not old_messages:
            return

        logger.info(
            "Summary trigger | total_msgs=%d total_chars=%d summarize_msgs=%d keep_msgs=%d",
            len(all_rows), total_chars, len(old_messages), len(all_rows) - split_idx,
        )

        summary_messages = self._build_summary_prompt_messages(
            conversation.summary,
            old_messages,
        )
        new_summary = self.llm_client.chat(summary_messages)
        update_conversation_summary(db, conversation.id, new_summary)

    @staticmethod
    def _drop_orphaned_user_messages(messages: list[dict]) -> list[dict]:
        """过滤被中止的 user 消息（无对应 assistant 回复）。
        仅保留完整的 user-assistant 对 + 最后一条 user（当前提问，尚未回复）。"""
        if not messages:
            return []

        result: list[dict] = []
        pending_user: dict | None = None

        for msg in messages:
            if msg.get("role") == "user":
                if pending_user is not None:
                    pass  # 前一条 user 没有 assistant 回复 → 被中止 → 丢弃
                pending_user = msg
            elif msg.get("role") == "assistant":
                if pending_user is not None:
                    result.append(pending_user)
                    result.append(msg)
                    pending_user = None
                # assistant 前面没有 user → 可能是被删除的遗留 → 丢弃
            # system 消息直接保留（summary 等）
            elif msg.get("role") == "system":
                if pending_user is not None:
                    # system 消息在 user 后面出现，先保留 user
                    result.append(pending_user)
                    pending_user = None
                result.append(msg)

        # 最后一条 user 消息（当前提问）保留
        if pending_user is not None:
            result.append(pending_user)

        dropped = len(messages) - len(result)
        if dropped:
            logger.info("Dropped %d orphaned user messages (aborted generations)", dropped)
        return result

    def _load_recent_context(self, db: Session, conversation) -> list[dict]:
        cached = None

        try:
            cached = get_cached_context_messages(conversation.conversation_uid)
        except Exception as exc:
            print(f"【Redis】读取上下文失败，降级 MySQL，会话={conversation.conversation_uid}, error={exc}")

        if cached is not None:
            print(f"【上下文读取】Redis HIT，会话={conversation.conversation_uid}，条数={len(cached)}")
            raw = [
                {"role": item["role"], "content": item["content"]}
                for item in cached
                if item.get("role") in {"user", "assistant", "system"}
            ]
            return self._drop_orphaned_user_messages(raw)

        print(f"【上下文读取】Redis MISS 或 Redis 不可用，回源 MySQL，会话={conversation.conversation_uid}")

        rows = list_recent_messages_by_conversation(
            db=db,
            conversation_id=conversation.id,
            limit=settings.chat_context_limit,
        )

        messages = [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
            if row.role in {"user", "assistant", "system"}
        ]

        try:
            self._safe_set_context_messages(conversation.conversation_uid, messages)
            print(f"【Redis】MySQL 回源后已回填 Redis，会话={conversation.conversation_uid}，条数={len(messages)}")
        except Exception as exc:
            print(f"【Redis】MySQL 回源成功，但回填 Redis 失败: {exc}")

        return self._drop_orphaned_user_messages([
            {"role": item["role"], "content": item["content"]}
            for item in messages
        ])

    def _build_llm_messages(
        self, conversation, recent_messages: list[dict], search_context: str = ""
    ) -> list[dict]:
        llm_messages = [{"role": "system", "content": _get_system_prompt()}]

        if conversation.summary:
            llm_messages.append(
                {
                    "role": "system",
                    "content": f"以下是该会话的历史摘要，请结合使用：\n{conversation.summary}",
                }
            )

        if search_context:
            llm_messages.append(
                {
                    "role": "system",
                    "content": f"以下是从项目文档中检索到的相关片段，请基于这些内容回答用户问题：\n\n{search_context}",
                }
            )

        llm_messages.extend(recent_messages)
        return llm_messages

    @staticmethod
    def _build_context_query(recent_messages: list[dict], current_query: str) -> str:
        """从最近对话历史提取上下文，生成带上下文的查询。用于提升短问题 / 指代问题的召回。

        仅使用最近 2 条用户消息作为指代消解上下文。
        不包含助手回复，避免 LLM 生成内容污染检索 embedding。"""
        user_queries: list[str] = []
        for msg in reversed(recent_messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # 跳过和当前问题相同的消息（去重）
                if content.strip() == current_query.strip():
                    continue
                user_queries.append(content.strip()[:80])
                if len(user_queries) >= 2:
                    break

        if not user_queries:
            return current_query

        user_queries.reverse()
        return "上文: " + " ; ".join(user_queries) + f" | 当前问题: {current_query}"

    @staticmethod
    def _should_expand_query(query: str) -> bool:
        """判断是否需要 Multi-Query 扩展。简单问题跳过，节省 LLM 调用。
        阈值 ~15 字，排除纯寒暄/闲聊。"""
        if len(query) < 15:
            return False
        simple_patterns = {"你好", "谢谢", "再见", "在吗", "hello", "hi", "ok", "好的", "是的", "对的", "嗯"}
        if query.strip().lower() in simple_patterns:
            return False
        return True

    def _expand_queries(self, query: str) -> list[str]:
        """Multi-Query: 用 LLM 生成 2-3 个不同视角的查询变体，提升召回覆盖面。"""
        if not self._should_expand_query(query):
            logger.info("Multi-Query skipped (simple query) | q=%.30s", query)
            return []
        try:
            messages = [{"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)}]
            resp = self.llm_client.chat(messages)
            variants = [q.strip() for q in resp.strip().split("\n") if q.strip() and q.strip() != query]
            # 去重 + 限制数量
            seen = {query}
            unique = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            if unique:
                logger.info("Multi-Query | original=%.40s → %d variants=%s", query, len(unique), unique[:3])
            return unique[:3]
        except Exception:
            logger.exception("Multi-Query expansion failed")
            return []

    def _hyde_expand(self, query: str, context: str = "") -> str:
        """HyDE: 让 LLM 生成假想文档片段。可选的 context 用于消歧短问题/指代问题。"""
        try:
            prompt = HYDE_PROMPT.format(query=query)
            if context:
                prompt = f"请先阅读对话历史来理解问题背景：\n{context}\n\n{HYDE_PROMPT.format(query=query)}"
            hyde_messages = [
                {"role": "user", "content": prompt},
            ]
            hypothetical = self.llm_client.chat(hyde_messages)
            logger.info("HyDE expanded | query_len=%d hyde_len=%d has_context=%s",
                        len(query), len(hypothetical), bool(context))
            return hypothetical.strip()
        except Exception:
            logger.exception("HyDE expansion failed, falling back to raw query")
            return ""

    def _do_search(self, db, query: str, recent_messages: list[dict] | None = None,
                   owner_id: int | None = None) -> tuple[list[dict], str]:
        """基于 SearchRouter 的 Workflow 级联检索。
        Simple/Factual/Analytical/Complex 自动路由。
        返回 (results, formatted_context, search_context) — 含路由信息供前端展示。
        """
        try:
            # 构建上下文增强查询
            context_query = ""
            if recent_messages:
                context_query = self._build_context_query(recent_messages, query)
                if context_query != query:
                    logger.info("Context query built | original=%.30s → contextual=%.50s", query, context_query)

            # 检索用上下文增强 query，分类用原始 query（避免上下文拼接干扰正则）
            search_query = context_query or query
            results, ctx = self.search_router.search(
                search_query, db, classify_query=query, owner_id=owner_id,
            )

            # 记录路由信息
            route_info = {
                "complexity": ctx.classification.complexity.value,
                "reason": ctx.classification.reason,
                "cache_hit": ctx.cache_hit,
                "timings": ctx.timings,
                "fusion": ctx.fusion_info,
            }
            logger.info(
                "Search routed | q=%.40s | route=%s cache=%s total=%.0fms",
                query, route_info["complexity"], route_info["cache_hit"],
                ctx.timings.get("total", 0),
            )
        except Exception:
            logger.exception("SearchRouter failed, falling back to hybrid search")
            try:
                results = self.hybrid_search.search(db, query, owner_id=owner_id)
                route_info = {"complexity": "fallback", "reason": "SearchRouter error"}
            except Exception:
                logger.exception("Hybrid search also failed")
                return [], "", {}

        context = self._format_search_context(results)
        return results, context, route_info

    def _try_scenario_match(self, db, query: str) -> list[dict]:
        """场景知识库预匹配。如果匹配成功，返回结构化知识条目列表。"""
        if not settings.scenario_kb_enabled:
            logger.debug("Scenario KB is disabled, skipping pre-match")
            return []
        try:
            from app.services.scenario_matcher import get_scenario_matcher
            matcher = get_scenario_matcher()
            result = matcher.match(
                db=db,
                query=query,
                top_k=settings.scenario_kb_max_results,
                threshold=settings.scenario_match_threshold,
            )
            entries = result.get("entries", [])
            if entries:
                logger.info(
                    "Scenario KB matched: %d entries (top_score=%.3f, elapsed=%.1fms)",
                    len(entries), entries[0].get("similarity_score", 0),
                    result.get("elapsed_ms", 0),
                )
            else:
                logger.debug("Scenario KB: no matches above threshold %.2f", settings.scenario_match_threshold)
            return entries
        except Exception:
            logger.warning("Scenario match failed, continuing without scenario KB")
            return []

    def _build_agent_search_fn(self, db, recent_messages: list[dict] | None = None,
                               original_query: str = "", owner_id: int | None = None):
        """构建 Agent 循环使用的检索函数。

        通过 SearchRouter.search() 走完整的搜索路由，包括：
        - 检索缓存
        - Query 分类 (QueryClassifier)
        - 路由到合适的检索策略 (simple/factual/analytical/complex)
        - DocSummaryIndex 文档级提升
        - 结果缓存

        上下文增强仅在 Round 1（原始用户查询）时应用，避免污染 Agent 工具
        生成的精确查询（rewrite_query/hyde_search 等）。
        """
        def search_fn(q: str) -> tuple[list[dict], dict]:
            # 仅对原始用户查询做上下文增强，工具生成的查询直接使用
            classify_query = None
            if recent_messages and original_query and q.strip() == original_query.strip():
                ctx_q = self._build_context_query(recent_messages, q)
                if ctx_q != q:
                    classify_query = q  # 用原始 query 做分类，避免上下文拼接干扰分类器
                    q = ctx_q

            t0 = time.time()
            results, ctx = self.search_router.search(
                q, db, classify_query=classify_query or q, owner_id=owner_id,
            )
            elapsed_ms = (time.time() - t0) * 1000

            fusion_info = ctx.fusion_info or {}
            if "faiss_ok" not in fusion_info:
                fusion_info["faiss_ok"] = bool(ctx.faiss_results)
            if "es_ok" not in fusion_info:
                fusion_info["es_ok"] = bool(ctx.es_results)

            logger.info(
                "Agent search_fn | q=%.50s -> %d results in %.0fms | "
                "cache_hit=%s complexity=%s faiss_ok=%s es_ok=%s conc=%.2f",
                q, len(results), elapsed_ms,
                ctx.cache_hit,
                ctx.classification.complexity.value if ctx.classification else "?",
                fusion_info.get("faiss_ok"), fusion_info.get("es_ok"),
                fusion_info.get("faiss_conc", fusion_info.get("es_conc", 0)),
            )
            return results, fusion_info
        return search_fn

    def _run_agent_loop(self, db, query: str, recent_messages: list[dict] | None = None,
                        owner_id: int | None = None, web_search: bool = False,
                        allow_web_search: bool = True):
        """阻塞式 Agent 循环（用于 chat）。返回 AgentState。"""
        search_fn = self._build_agent_search_fn(db, recent_messages, original_query=query, owner_id=owner_id)

        # ── 场景知识库预匹配 ──
        scenario_matches = self._try_scenario_match(db, query)

        try:
            # 多Agent编排：run_agent_loop 内部从 ModelRegistry 获取各角色模型
            agent_state = run_agent_loop(
                user_query=query,
                search_fn=search_fn,
                progress_callback=None,
                force_web_search=web_search,
                allow_web_search=allow_web_search,
            )
            agent_state.scenario_matches = scenario_matches
            if scenario_matches:
                agent_state.scenario_match_score = scenario_matches[0]["similarity_score"]
                # 如果文档检索触发了降级，但场景知识库有匹配 → 重置降级，用场景知识库回答
                if agent_state.degradation_triggered:
                    logger.info(
                        "Scenario KB has %d matches, overriding document-search degradation",
                        len(scenario_matches),
                    )
                    agent_state.degradation_triggered = False
                    agent_state.degradation_result = None
                    agent_state.final_answer = ""
                    agent_state.suppress_sources = False  # 场景KB有匹配，允许展示来源
            # 最终回答生成（Generator Agent, pro 模型）
            if not agent_state.final_answer:
                agent_state.final_answer = _generate_final_answer(
                    self.generator_client, agent_state, query,
                    agent_state.all_results if agent_state.all_results else agent_state.search_results,
                )
        except Exception:
            logger.exception("Agent loop failed, falling back to legacy search")
            results, search_context, route_info = self._do_search(db, query, recent_messages, owner_id=owner_id)
            llm_messages = self._build_llm_messages(None, recent_messages or [], search_context)
            answer = self.llm_client.chat(llm_messages)
            agent_state = AgentState(
                user_query=query,
                classification=ClassificationResult(
                    query_class=QueryClass.SHALLOW, confidence=0.3,
                    reason="fallback", max_rounds=1, max_tool_calls_per_round=1,
                    allowed_tool_categories=["search"], degradation_max_level=1,
                ),
                round=1, max_rounds=1,
                search_results=results, all_results=results,
                final_answer=answer,
                fusion_info=route_info.get("fusion"),
                scenario_matches=scenario_matches,
            )

        return agent_state

    @staticmethod
    def _build_usage_metadata(state: AgentState, sources: list[SourceReference] | None = None) -> dict:
        """Build UI-facing execution metadata from the real agent state."""
        all_results = getattr(state, "all_results", None) or getattr(state, "search_results", None) or []
        source_items = sources or []
        tool_history = getattr(state, "tool_call_history", None) or []
        effective_tool_history = [
            tc for tc in tool_history
            if not (
                tc.get("name") == "web_search"
                and "未启用" in str(tc.get("result_summary", ""))
            )
        ]
        tool_names = [tc.get("name", "") for tc in effective_tool_history if tc.get("name")]
        web_tool_used = any(
            tc.get("name") == "web_search"
            and "未启用" not in str(tc.get("result_summary", ""))
            for tc in effective_tool_history
        )
        result_doc_uids = {r.get("document_uid", "") for r in all_results}
        source_doc_uids = {getattr(s, "document_uid", "") for s in source_items}
        degradation_result = getattr(state, "degradation_result", None)
        degradation_used_levels = getattr(degradation_result, "used_levels", []) if degradation_result else []
        answer = getattr(state, "final_answer", "") or ""

        def _web_source(title: str = "", url: str = "", summary: str = "", domain: str = "") -> dict | None:
            title = (title or "").strip()
            url = (url or "").strip()
            summary = (summary or "").strip()
            if not title and not url and not summary:
                return None
            if not title:
                title = url or "联网搜索来源"
            return {
                "title": title,
                "url": url,
                "domain": domain.strip() if isinstance(domain, str) else "",
                "summary": summary[:300],
            }

        web_sources: list[dict] = []

        def _add_web_source(item: dict | None):
            if not item:
                return
            key = item.get("url") or item.get("title")
            if key and any((s.get("url") or s.get("title")) == key for s in web_sources):
                return
            web_sources.append(item)

        for r in all_results:
            if r.get("document_uid") == "web_search":
                _add_web_source(_web_source(
                    title=r.get("filename") or r.get("title", ""),
                    url=r.get("url") or r.get("section_title", ""),
                    summary=r.get("content") or r.get("snippet", ""),
                    domain=r.get("domain", ""),
                ))
        for s in source_items:
            if getattr(s, "document_uid", "") == "web_search":
                _add_web_source(_web_source(
                    title=getattr(s, "filename", ""),
                    url=getattr(s, "section_title", ""),
                    summary=getattr(s, "content_preview", ""),
                ))
        if degradation_result:
            for wr in getattr(degradation_result, "level2_web_results", []) or []:
                _add_web_source(_web_source(
                    title=wr.get("title", ""),
                    url=wr.get("url", ""),
                    summary=wr.get("snippet") or wr.get("summary") or wr.get("content", ""),
                    domain=wr.get("domain", ""),
                ))

        used_web_search = (
            web_tool_used
            or "web_search" in result_doc_uids
            or "web_search" in source_doc_uids
            or 2 in degradation_used_levels
            or bool(web_sources)
        )
        local_doc_uids = {
            uid for uid in (result_doc_uids | source_doc_uids)
            if uid and uid not in {"web_search", "scenario_kb", "doc_gen", "diagram_gen"}
        }
        used_scenario = bool(getattr(state, "scenario_matches", None))
        search_tool_names = {
            "search_knowledge_base",
            "search_scenario_kb",
            "rewrite_query",
            "multi_query_search",
            "hyde_search",
            "rerank_results",
        }
        retrieval_attempted = bool(local_doc_uids or used_scenario or any(n in search_tool_names for n in tool_names))
        used_general = "[通用知识]" in answer or (not used_web_search and not local_doc_uids and not used_scenario)
        general_reason = ""
        if used_general and retrieval_attempted:
            general_reason = "场景知识库或知识库命中不足，已使用通用知识补充。"
        elif used_general:
            general_reason = "未命中可展示的外部来源，回答基于通用知识生成。"

        return {
            "used_web_search": used_web_search,
            "used_rag": bool(local_doc_uids or used_scenario),
            "used_scenario_kb": used_scenario,
            "used_tools": bool(tool_names),
            "retrieval_attempted": retrieval_attempted,
            "tool_names": sorted(set(tool_names)),
            "source_count": len(source_items),
            "result_count": len(all_results),
            "web_sources": web_sources,
            "used_general_knowledge": used_general,
            "general_reason": general_reason,
        }

    def _run_agent_loop_stream(self, db, query: str, recent_messages: list[dict] | None = None,
                               state_container: list | None = None, owner_id: int | None = None,
                               web_search: bool = False, allow_web_search: bool = True,
                               model_config: dict | None = None):
        """流式 Agent 循环生成器（用于 stream_chat）。yield SSE 事件。

        Args:
            state_container: 可变容器 [None]，完成后写入 agent_state。
        """
        import queue
        import threading

        event_queue = queue.Queue()
        holder = state_container if state_container is not None else [None]

        search_fn = self._build_agent_search_fn(db, recent_messages, original_query=query, owner_id=owner_id)

        # ── 场景知识库预匹配 ──
        scenario_matches = self._try_scenario_match(db, query)

        def _progress_callback(event: dict):
            event_queue.put(event)

        def _run_in_thread():
            try:
                state = run_agent_loop(
                    user_query=query,
                    search_fn=search_fn,
                    progress_callback=_progress_callback,
                    force_web_search=web_search,
                    allow_web_search=allow_web_search,
                    model_overrides=model_config,
                )
                state.scenario_matches = scenario_matches
                if scenario_matches:
                    state.scenario_match_score = scenario_matches[0]["similarity_score"]
                    # 文档检索触发降级但场景知识库有匹配 → 重置降级
                    if state.degradation_triggered:
                        logger.info(
                            "Scenario KB has %d matches, overriding document-search degradation (stream)",
                            len(scenario_matches),
                        )
                        state.degradation_triggered = False
                        state.degradation_result = None
                        state.final_answer = ""
                holder[0] = state
                event_queue.put({"type": "_agent_done", "state": state})
            except Exception as e:
                logger.exception("Agent loop thread failed")
                event_queue.put({"type": "_agent_error", "error": str(e)})

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            event_type = event.get("type", "")

            if event_type == "_agent_done":
                state = event.get("state")
                if state:
                    # 场景知识库匹配结果（结构化知识卡片）
                    if state.scenario_matches:
                        yield {"type": "scenario_matches", "entries": state.scenario_matches}
                    yield {"type": "status", "content": f"检索完成，共 {len(state.all_results)} 条结果，正在生成回答..."}
                    # 场景知识库高置信度匹配时，过滤低相关度文档结果
                    # 场景知识库无匹配时，基于检索信号质量过滤
                    results_for_answer = state.all_results
                    if state.scenario_matches and state.scenario_match_score >= 0.6:
                        from app.agent.agent_loop import _filter_results_by_scenario_context
                        results_for_answer = _filter_results_by_scenario_context(
                            state.all_results, state,
                        )
                        # 场景过滤移除全部本地文档 → 本地文档与此问题无关，抑制来源
                        if not results_for_answer and len(state.all_results) > 0:
                            state.suppress_sources = True
                            logger.info(
                                "Scenario filter removed all %d local doc results "
                                "(scenario=%.40s), suppressing irrelevant sources",
                                len(state.all_results),
                                state.scenario_matches[0].get("title", "") if state.scenario_matches else "",
                            )
                    else:
                        from app.agent.agent_loop import _filter_results_by_quality
                        results_for_answer = _filter_results_by_quality(
                            state.all_results, state,
                        )
                    # 先流式输出最终回答
                    full = ""
                    for token in _generate_final_answer_stream(
                        self.generator_client, state, query, results_for_answer,
                    ):
                        full += token
                        yield token
                    state.final_answer = full
                    # 回答生成后再发 sources（此时 suppress_sources 已确定）
                    raw_sources = []
                    if not state.suppress_sources:
                        raw_sources = self._build_sources(state.all_results)
                        yield {"type": "sources", "sources": [s.model_dump() for s in raw_sources]}
                    yield {"type": "usage_metadata", "metadata": self._build_usage_metadata(state, raw_sources)}
                    holder[0] = state
                break
            elif event_type == "_agent_error":
                error_msg = event.get("error", "未知错误")
                yield {"type": "error", "content": f"Agent 执行失败: {error_msg}"}
                break
            elif event_type == "agent_status":
                yield event

        thread.join(timeout=5)
        return holder[0]

    def _format_search_context(self, results: list[dict]) -> str:
        """将搜索结果格式化为 LLM 可读的上下文字符串。
        关键：内容与元数据严格分离。正文只给纯内容+编号，元数据放在末尾附录。
        LLM 只看到纯内容，无法在正文中插入行内引用。
        参考来源卡片由前端 renderSources 统一渲染。"""
        if not results:
            return ""
        fragments = []
        appendix = []
        for i, r in enumerate(results, 1):
            fragments.append(f"[片段 {i}]\n{r['content']}")
            meta_parts = [f"片段 {i}"]
            if r.get("filename"):
                meta_parts.append(r["filename"])
            if r.get("section_title"):
                meta_parts.append(r["section_title"])
            if r.get("page_no"):
                meta_parts.append(f"第{r['page_no']}页")
            appendix.append("  ·  ".join(meta_parts))

        body = "\n\n---\n\n".join(fragments)
        footer = "\n".join(appendix)
        return f"{body}\n\n---\n检索信息索引（仅供你了解来源，不要在正文中引用）:\n{footer}"

    def _build_sources(self, results: list[dict]) -> list[SourceReference]:
        """构建 API 响应中的来源列表。"""
        return [
            SourceReference(
                chunk_uid=r["chunk_uid"],
                document_uid=r["document_uid"],
                filename=r.get("filename", ""),
                content_preview=(r.get("content", "") or "")[:200],
                section_title=r.get("section_title", ""),
                page_no=r.get("page_no", 0),
                score=round(r.get("rrf_score", r.get("faiss_score", 0)), 4),
            )
            for r in results[:10]
        ]

    @staticmethod
    def _sources_to_json(sources: list[SourceReference]) -> str | None:
        """序列化来源列表为 JSON 字符串，用于持久化到 DB。"""
        if not sources:
            return None
        return json.dumps([s.model_dump() for s in sources], ensure_ascii=False)

    @staticmethod
    def _parse_sources_json(sources_json: str | None) -> list[SourceReference]:
        """从 DB 的 JSON 字符串反序列化来源列表。"""
        if not sources_json:
            return []
        try:
            data = json.loads(sources_json)
            return [SourceReference(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _parse_metadata_json(metadata_json: str | None) -> dict | None:
        """从 DB 的 JSON 字符串反序列化使用情况 metadata。"""
        if not metadata_json:
            return None
        try:
            return json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            return None

    def _invalidate_conversation_cache(
            self,
            db: Session,
            conversation_uid: str,
            reason: str,
    ):
        delete_cache_with_retry_task(
            db=db,
            conversation_uid=conversation_uid,
            reason=reason,
        )

    def delete_conversation(self, db: Session, conversation_id: str, user=None):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        ok = delete_conversation_and_messages(db, conversation_id, owner_id=owner_id)

        if ok:
            try:
                delete_conversation_cache(conversation_id)
            except Exception:
                delete_cache_with_retry_task(
                    db=db,
                    conversation_uid=conversation_id,
                    reason="delete_conversation",
                )

        return {
            "ok": ok,
            "message": "会话已删除" if ok else "删除失败",
        }

    def delete_messages(
            self,
            db: Session,
            conversation_id: str,
            message_ids: list[int],
            user=None,
    ):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        deleted = delete_messages_by_ids(db, message_ids)

        touch_conversation(db, conversation.id)

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="delete_message",
        )

        return {
            "ok": True,
            "message": f"已删除 {deleted} 条消息",
        }

    def regenerate_last_answer(self, db: Session, conversation_id: str, user=None):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        last_assistant = get_last_assistant_message(db, conversation.id)

        if last_assistant is None:
            raise ValueError("没有可重新生成的助手回复")

        db.delete(last_assistant)
        db.commit()

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="regenerate_answer",
        )

        last_user = get_last_user_message(db, conversation.id)
        user_query = last_user.content if last_user else ""

        recent_messages = self._load_recent_context(db, conversation)
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        agent_state = self._run_agent_loop(
            db, user_query, recent_messages,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        answer = agent_state.final_answer

        sources = [] if agent_state.suppress_sources else self._build_sources(
            agent_state.all_results if agent_state.all_results else agent_state.search_results
        )
        assistant_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources_json=self._sources_to_json(sources),
        )

        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=answer,
                created_at=assistant_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】重新生成后追加 assistant 缓存失败，不影响主流程: {exc}")

        touch_conversation(db, conversation.id)

        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid,
            title=conversation.title,
        )

        return {
            "conversation_id": conversation.conversation_uid,
            "answer": answer,
            "sources": sources,
        }

    def edit_last_user_and_regenerate(
            self,
            db: Session,
            conversation_id: str,
            new_content: str,
            user=None,
    ):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        last_user = get_last_user_message(db, conversation.id)

        if last_user is None:
            raise ValueError("没有可编辑的用户问题")

        update_message_content(db, last_user.id, new_content)

        delete_messages_after_id(
            db=db,
            conversation_id=conversation.id,
            message_id=last_user.id,
        )

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="edit_message",
        )

        recent_messages = self._load_recent_context(db, conversation)
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        agent_state = self._run_agent_loop(
            db, new_content, recent_messages,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        answer = agent_state.final_answer

        sources = [] if agent_state.suppress_sources else self._build_sources(
            agent_state.all_results if agent_state.all_results else agent_state.search_results
        )
        assistant_row = create_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources_json=self._sources_to_json(sources),
        )

        try:
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=answer,
                created_at=assistant_row.created_at,
            )
        except Exception as exc:
            print(f"【Redis】编辑重发后追加 assistant 缓存失败，不影响主流程: {exc}")

        touch_conversation(db, conversation.id)

        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid,
            title=conversation.title,
        )

        return {
            "conversation_id": conversation.conversation_uid,
            "answer": answer,
            "sources": sources,
        }

    def stream_regenerate_last_answer(self, db: Session, conversation_id: str, user=None):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        last_assistant = get_last_assistant_message(db, conversation.id)

        if last_assistant is None:
            raise ValueError("没有可重新生成的助手回复")

        db.delete(last_assistant)
        db.commit()

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="regenerate_answer",
        )

        last_user = get_last_user_message(db, conversation.id)
        user_query = last_user.content if last_user else ""

        recent_messages = self._load_recent_context(db, conversation)

        yield {"type": "meta", "conversation_id": conversation.conversation_uid}

        # 走 Agent 循环（流式）
        holder = [None]
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        yield from self._run_agent_loop_stream(
            db, user_query, recent_messages,
            state_container=holder,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        agent_state = holder[0]

        # 流结束后保存新的 assistant
        if agent_state and agent_state.final_answer:
            sources = [] if agent_state.suppress_sources else self._build_sources(
                agent_state.all_results if agent_state.all_results else agent_state.search_results
            )
            assistant_row = create_message(
                db=db,
                conversation_id=conversation.id,
                role="assistant",
                content=agent_state.final_answer,
                sources_json=self._sources_to_json(sources),
            )

            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=agent_state.final_answer,
                created_at=assistant_row.created_at,
            )

        touch_conversation(db, conversation.id)

        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid,
            title=conversation.title,
        )

    def stream_edit_last_user_and_regenerate(
            self,
            db: Session,
            conversation_id: str,
            new_content: str,
            user=None,
    ):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)

        if conversation is None:
            raise ValueError("会话不存在")

        last_user = get_last_user_message(db, conversation.id)

        if last_user is None:
            raise ValueError("没有可编辑的用户问题")

        update_message_content(db, last_user.id, new_content)

        delete_messages_after_id(
            db=db,
            conversation_id=conversation.id,
            message_id=last_user.id,
        )

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="edit_message",
        )

        recent_messages = self._load_recent_context(db, conversation)

        yield {"type": "meta", "conversation_id": conversation.conversation_uid}

        # 走 Agent 循环（流式）
        holder = [None]
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        yield from self._run_agent_loop_stream(
            db, new_content, recent_messages,
            state_container=holder,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        agent_state = holder[0]

        if agent_state and agent_state.final_answer:
            sources = [] if agent_state.suppress_sources else self._build_sources(
                agent_state.all_results if agent_state.all_results else agent_state.search_results
            )
            assistant_row = create_message(
                db=db,
                conversation_id=conversation.id,
                role="assistant",
                content=agent_state.final_answer,
                sources_json=self._sources_to_json(sources),
            )

            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant",
                content=agent_state.final_answer,
                created_at=assistant_row.created_at,
            )

        touch_conversation(db, conversation.id)

        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid,
            title=conversation.title,
        )

    def edit_message_and_regenerate(
            self,
            db: Session,
            conversation_id: str,
            message_id: int,
            new_content: str,
            user=None,
    ):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)
        if conversation is None:
            raise ValueError("会话不存在")

        msg = get_message_by_id(db, message_id)
        if msg is None or msg.conversation_id != conversation.id or msg.role != "user":
            raise ValueError("消息不存在或不可编辑")

        update_message_content(db, message_id, new_content)
        self._delete_next_assistant(db, conversation.id, message_id)

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="edit_message",
        )

        recent_messages = self._load_recent_context(db, conversation)
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        agent_state = self._run_agent_loop(
            db, new_content, recent_messages,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        answer = agent_state.final_answer

        sources = [] if agent_state.suppress_sources else self._build_sources(
            agent_state.all_results if agent_state.all_results else agent_state.search_results
        )
        assistant_row = create_message(
            db=db, conversation_id=conversation.id, role="assistant", content=answer,
            sources_json=self._sources_to_json(sources),
        )

        self._safe_append_context_message(
            conversation_uid=conversation.conversation_uid,
            role="assistant", content=answer, created_at=assistant_row.created_at,
        )
        touch_conversation(db, conversation.id)
        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid, title=conversation.title,
        )

        return {
            "conversation_id": conversation.conversation_uid,
            "answer": answer,
            "sources": sources,
        }

    @staticmethod
    def _delete_next_assistant(db: Session, conversation_id: int, after_message_id: int) -> None:
        """删除指定 user 消息后面紧接的那条 assistant 回复（不删后续对话）。"""
        from app.models.message import Message as MsgModel
        next_msg = (
            db.query(MsgModel)
            .filter(
                MsgModel.conversation_id == conversation_id,
                MsgModel.id > after_message_id,
            )
            .order_by(MsgModel.id.asc())
            .first()
        )
        if next_msg is not None and next_msg.role == "assistant":
            db.delete(next_msg)
            db.commit()

    def stream_edit_message_and_regenerate(
            self,
            db: Session,
            conversation_id: str,
            message_id: int,
            new_content: str,
            user=None,
    ):
        owner_id = None if (user and getattr(user, 'role', None) == 'admin') else (user.id if user else None)
        conversation = get_by_uid(db, conversation_id, owner_id=owner_id)
        if conversation is None:
            raise ValueError("会话不存在")

        msg = get_message_by_id(db, message_id)
        if msg is None or msg.conversation_id != conversation.id or msg.role != "user":
            raise ValueError("消息不存在或不可编辑")

        update_message_content(db, message_id, new_content)
        self._delete_next_assistant(db, conversation.id, message_id)

        self._invalidate_conversation_cache(
            db=db,
            conversation_uid=conversation.conversation_uid,
            reason="edit_message",
        )

        recent_messages = self._load_recent_context(db, conversation)

        yield {"type": "meta", "conversation_id": conversation.conversation_uid}

        # 走 Agent 循环（流式）
        holder = [None]
        allow_web_search = not (user and getattr(user, "role", None) == "guest")
        yield from self._run_agent_loop_stream(
            db, new_content, recent_messages,
            state_container=holder,
            owner_id=owner_id,
            allow_web_search=allow_web_search,
        )
        agent_state = holder[0]

        if agent_state and agent_state.final_answer:
            sources = [] if agent_state.suppress_sources else self._build_sources(
                agent_state.all_results if agent_state.all_results else agent_state.search_results
            )
            assistant_row = create_message(
                db=db, conversation_id=conversation.id, role="assistant", content=agent_state.final_answer,
                sources_json=self._sources_to_json(sources),
            )
            self._safe_append_context_message(
                conversation_uid=conversation.conversation_uid,
                role="assistant", content=agent_state.final_answer, created_at=assistant_row.created_at,
            )

        touch_conversation(db, conversation.id)
        self._safe_touch_recent_conversation(
            conversation_uid=conversation.conversation_uid, title=conversation.title,
        )
