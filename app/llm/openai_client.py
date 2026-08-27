from __future__ import annotations

import json
import logging

import httpx
from app.core.config import get_settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.openai_model or "gpt-4o-mini"
        self._api_key = api_key
        self._base_url = base_url

        logger.info("LLM client init | model=%s | base_url=%s | mock=%s",
                     self.model, base_url or self.settings.openai_base_url or "(default)", self.settings.mock_llm)

        if self.settings.mock_llm:
            self.client = None
        else:
            client_kwargs = {
                "api_key": self._api_key or self.settings.openai_api_key,
                "timeout": 60.0,
                "max_retries": 1,
            }
            effective_url = self._base_url or self.settings.openai_base_url
            if effective_url:
                client_kwargs["base_url"] = effective_url

            # trust_env=False 防止读取 HTTP_PROXY/HTTPS_PROXY 环境变量，
            # 避免 Clash 代理干扰直连 API 的 SSL 握手
            http_client = httpx.Client(trust_env=False)
            client_kwargs["http_client"] = http_client
            self.client = OpenAI(**client_kwargs)

    # ── 普通调用（非流式） ──

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        if self.settings.mock_llm:
            latest_user_message = next(
                (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
                "",
            )
            history_count = sum(
                1 for msg in messages if msg["role"] in {"user", "assistant"}
            )
            return (
                "【本地Mock回复】\n"
                f"我已收到你的问题：{latest_user_message}\n"
                f"当前会话已累计 {history_count} 条消息。\n"
            )

        if self.client is None:
            raise RuntimeError("OpenAI client 未初始化")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content or "模型没有返回内容"

    # ── Function Calling（非流式） ──

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.7,
    ) -> dict:
        """支持 function calling 的对话调用。

        返回:
            {
                "content": str | None,       # 纯文本回复（可能为 None）
                "tool_calls": list[dict] | None,  # 工具调用列表
                "finish_reason": str,
            }
        """
        if self.settings.mock_llm:
            return {
                "content": "【Mock】Function calling 模式下的模拟回复",
                "tool_calls": None,
                "finish_reason": "stop",
            }

        if self.client is None:
            raise RuntimeError("OpenAI client 未初始化")

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        result: dict = {
            "content": message.content,
            "tool_calls": None,
            "finish_reason": choice.finish_reason,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result

    # ── 流式调用（SSE 核心） ──

    def stream_chat(self, messages: list[dict], temperature: float = 0.7):
        if self.settings.mock_llm:
            yield "【Mock流式】你好，这是模拟流式输出。"
            yield "当前系统已经支持 SSE 流式能力。"
            return

        if self.client is None:
            raise RuntimeError("OpenAI client 未初始化")

        logger.info("LLM stream start | model=%s | messages=%d", self.model, len(messages))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=temperature,
        )

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if not hasattr(choice, "delta"):
                continue
            delta = choice.delta
            if not delta:
                continue
            content = delta.content
            if content:
                yield content

        logger.info("LLM stream complete")

    # ── Function Calling 流式 ──

    def stream_chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.7,
    ):
        """支持 function calling 的流式对话。

        Yields:
            {"type": "token", "content": str}   — 文本 token
            {"type": "tool_call_start"}           — 工具调用开始
            {"type": "tool_call_delta", "id": str, "name": str, "arguments": str}  — 工具调用增量
            {"type": "done", "finish_reason": str} — 结束
        """
        if self.settings.mock_llm:
            yield {"type": "token", "content": "【Mock流式·Function Calling】"}
            yield {"type": "token", "content": "这是模拟的流式输出。"}
            yield {"type": "done", "finish_reason": "stop"}
            return

        if self.client is None:
            raise RuntimeError("OpenAI client 未初始化")

        logger.info("LLM stream+tool start | model=%s | messages=%d | tools=%d",
                     self.model, len(messages), len(tools))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

        current_tool_call_id: str | None = None
        current_tool_name: str | None = None

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if not delta:
                continue

            # 文本 token
            if delta.content:
                yield {"type": "token", "content": delta.content}

            # 工具调用
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.id:
                        current_tool_call_id = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_name = tc_delta.function.name
                            yield {
                                "type": "tool_call_start",
                                "id": current_tool_call_id,
                                "name": current_tool_name,
                            }
                        if tc_delta.function.arguments:
                            yield {
                                "type": "tool_call_delta",
                                "id": current_tool_call_id,
                                "name": current_tool_name,
                                "arguments": tc_delta.function.arguments,
                            }

            if choice.finish_reason:
                yield {"type": "done", "finish_reason": choice.finish_reason}

        logger.info("LLM stream+tool complete")

    # ── JSON 模式（用于结构化输出如分类、Critic） ──

    def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> dict | None:
        """要求 LLM 输出 JSON，自动解析返回。失败返回 None。"""
        messages = list(messages)
        # 在最后一条消息末尾追加 JSON 格式要求
        if messages and messages[-1]["role"] == "user":
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] += "\n\n请只输出 JSON，不要包含其他内容。"
        else:
            messages.append({
                "role": "user",
                "content": "请只输出 JSON，不要包含其他内容。",
            })

        try:
            raw = self.chat(messages, temperature=temperature)
            # 尝试提取 JSON（处理 ```json ... ``` 包裹的情况）
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                raw = "\n".join(lines)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            logger.exception("Failed to parse JSON from LLM response")
            return None
