"""ModelRegistry — 多 Agent 架构的模型客户端注册中心。

每个 Agent 角色可独立配置模型，空则回退到默认模型。
典型配置（DeepSeek 系列）：
  - classifier:    deepseek-v4-flash  (轻量)
  - evaluator:     deepseek-v4-flash  (轻量)
  - generator:     deepseek-v4-pro    (核心输出，最强)
  - doc_generator: deepseek-v4-flash  (格式化，轻量)
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.llm.openai_client import OpenAIChatClient

logger = logging.getLogger(__name__)

# 角色 → 配置字段 映射
_ROLE_CONFIG_KEYS = {
    "classifier": "classifier_model",
    "evaluator": "evaluator_model",
    "generator": "generator_model",
    "doc_generator": "doc_generator_model",
}


class ModelRegistry:
    """多模型客户端注册中心（单例）。"""

    def __init__(self):
        self._clients: dict[str, OpenAIChatClient] = {}

    def get_client(self, role: str, overrides: dict | None = None) -> OpenAIChatClient:
        """获取指定角色的模型客户端。按需创建，缓存复用。

        overrides: 客户端传来的运行时配置，格式如:
          {"generator": {"model": "deepseek-v4-pro", "api_key": "sk-...", "base_url": "https://..."}}
        当 overrides 包含当前角色时，创建临时客户端（不使用缓存）。
        """
        if role not in _ROLE_CONFIG_KEYS:
            raise ValueError(f"Unknown model role: {role}. Valid: {list(_ROLE_CONFIG_KEYS)}")

        role_override = overrides.get(role) if overrides else None
        if role_override and role_override.get("model"):
            # 客户端自定义配置 → 创建临时客户端
            client = OpenAIChatClient(
                model=role_override["model"],
                api_key=role_override.get("api_key") or None,
                base_url=role_override.get("base_url") or None,
            )
            logger.info("ModelRegistry: role=%s override model=%s (not cached)", role, client.model)
            return client

        if role not in self._clients:
            config_key = _ROLE_CONFIG_KEYS[role]
            model_name = getattr(settings, config_key, "") or None
            client = OpenAIChatClient(model=model_name)
            self._clients[role] = client
            logger.info("ModelRegistry: role=%s model=%s", role, client.model)

        return self._clients[role]

    def get_default_client(self) -> OpenAIChatClient:
        """获取默认模型客户端（兼容旧接口）。"""
        return self.get_client("generator")

    def reset(self):
        """清空缓存（配置变更后调用）。"""
        self._clients.clear()


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
