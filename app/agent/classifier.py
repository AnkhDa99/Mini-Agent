"""
LLM 问题分类器：chitchat / shallow / deep。

- chitchat: 问候、感谢、闲聊，与企业技术内容无关 → 跳过检索
- shallow:  单一概念解释、简单对比、有明确答案 → 最多 2 轮，禁用重型工具
- deep:    需要综合多文档、分析推理、生成文档 → 最多 5 轮，全部工具可用

使用轻量 LLM 调用（~300ms），比规则分类更准确。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryClass(Enum):
    CHITCHAT = "chitchat"
    SHALLOW = "shallow"
    DEEP = "deep"


@dataclass
class ClassificationResult:
    query_class: QueryClass
    confidence: float
    reason: str
    max_rounds: int = 1
    max_tool_calls_per_round: int = 1
    allowed_tool_categories: list[str] = field(default_factory=list)
    # 降级链级别: shallow 只到 Level 1, deep 到 Level 2
    degradation_max_level: int = 1


# 预检正则：极端明显的闲聊不需要 LLM，节省 300ms
CHITCHAT_PATTERNS = [
    r'^(你好|您好|hi|hello|hey|早上好|下午好|晚上好)\s*$',
    r'^(谢谢|多谢|感谢|thanks|thank you|thx)\s*$',
    r'^(再见|拜拜|bye|goodbye|see you|回头见)\s*$',
    r'^(嗯|哦|好|ok|okay|是的|对的|好的|行|可以|知道了|明白了)\s*$',
    r'^(哈哈|呵呵|嘿嘿|嘻嘻|haha|lol)\s*$',
    r'^(在吗|在不在|你叫什么|你是谁|你是什么|你是干嘛的)\s*$',
]


def _regex_precheck(query: str) -> ClassificationResult | None:
    """极端简单的闲聊直接用正则拦截，不调 LLM。"""
    import re
    q = query.strip().lower()
    for pattern in CHITCHAT_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            return ClassificationResult(
                query_class=QueryClass.CHITCHAT,
                confidence=1.0,
                reason="regex: trivial chitchat",
                max_rounds=1,
                max_tool_calls_per_round=0,
                allowed_tool_categories=[],
                degradation_max_level=0,
            )
    return None


CLASSIFIER_SYSTEM_PROMPT = """你是一个问题复杂度分类器。分析用户问题，判断其深度。

分类标准：
- chitchat: 问候、感谢、天气、闲聊、和项目技术内容无关的话题
- shallow: 有明确答案的单一问题。单一概念解释、简单对比、列举已知项、参数查询。
           示例："什么是Linux？"、"Redis有哪些数据类型？"、"Java和Python的区别是什么？"
- deep: 需要综合多文档、分析推理、方案设计、生成新文档（PPT/Word/报告）。
         示例："为什么选择React而不是Vue？"、"整理XX系统的架构设计并生成PPT"、
         "我们项目的并发编程方案有什么风险？"

重要规则：
1. 如果问题与企业技术/项目文档相关但只需一个明确答案 → shallow
2. 如果问题需要综合多份文档或需要分析推理 → deep
3. 涉及"生成文档"、"制作PPT"、"写报告" → 一定是 deep

输出格式（严格 JSON）：
{"class": "chitchat|shallow|deep", "reason": "一句话理由"}"""


def classify(
    query: str,
    llm_client=None,
) -> ClassificationResult:
    """主入口：预检正则 → LLM 分类 → 返回结构化结果。"""
    # 正则预检
    regex_result = _regex_precheck(query)
    if regex_result:
        logger.info("Classifier (regex) | q=%.50s → %s", query, regex_result.query_class.value)
        return regex_result

    if llm_client is None:
        # 无 LLM：兜底为 shallow（保守，至少检索 1 次）
        logger.warning("Classifier: no LLM client, defaulting to shallow")
        return ClassificationResult(
            query_class=QueryClass.SHALLOW,
            confidence=0.3,
            reason="no LLM available, default shallow",
            max_rounds=2,
            max_tool_calls_per_round=1,
            allowed_tool_categories=["search"],
            degradation_max_level=1,
        )

    # LLM 分类
    try:
        result = llm_client.chat_json([
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ], temperature=0.1)

        if not result:
            raise ValueError("LLM returned empty JSON")

        class_str = result.get("class", "shallow").strip().lower()
        reason = result.get("reason", "LLM classified")

        if class_str not in ("chitchat", "shallow", "deep"):
            class_str = "shallow"

        query_class = QueryClass(class_str)

    except Exception:
        logger.exception("LLM classification failed, defaulting to shallow")
        query_class = QueryClass.SHALLOW
        reason = "LLM classification failed, default shallow"

    logger.info("Classifier (LLM) | q=%.60s → %s (%.2f) %s",
                query, query_class.value, 0.85, reason)

    if query_class == QueryClass.CHITCHAT:
        return ClassificationResult(
            query_class=QueryClass.CHITCHAT,
            confidence=0.9,
            reason=reason,
            max_rounds=1,
            max_tool_calls_per_round=0,
            allowed_tool_categories=[],
            degradation_max_level=0,
        )

    if query_class == QueryClass.SHALLOW:
        return ClassificationResult(
            query_class=QueryClass.SHALLOW,
            confidence=0.85,
            reason=reason,
            max_rounds=2,
            max_tool_calls_per_round=1,
            allowed_tool_categories=["search", "mcp"],
            degradation_max_level=2,
        )

    # deep
    return ClassificationResult(
        query_class=QueryClass.DEEP,
        confidence=0.85,
        reason=reason,
        max_rounds=5,
        max_tool_calls_per_round=2,
        allowed_tool_categories=["search", "analysis", "generation", "mcp"],
        degradation_max_level=2,
    )
