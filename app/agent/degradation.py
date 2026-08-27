"""
降级链：文档库完全无结果时的诚实回答策略。

三级降级：
  Level 0: 诚实声明 "文档库中未找到相关内容"（必须）
  Level 1: LLM 自身知识 + [通用知识] 标注
  Level 2: MCP 联网搜索 + [网络搜索] 标注（SHALLOW/DEEP 均可触发）

"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DegradationResult:
    """降级链的输出。"""
    level0_declaration: str  # 诚实声明
    level1_answer: str = ""  # 模型知识回答（带 [通用知识] 标注）
    level2_web_results: list[dict] = field(default_factory=list)  # 联网搜索原始结果
    level2_integrated_answer: str = ""  # LLM 整合网络搜索结果后的回答
    used_levels: list[int] = field(default_factory=list)  # 实际使用的降级级别
    sources_labeled: bool = False  # 是否已标注来源


DEGRADATION_SYSTEM_PROMPT = """你是 Mini Agent，一个项目记忆与协作助手。

**重要：项目文档库中未找到与用户问题相关的内容。** 这很正常——用户可能切换了话题，或者文档库暂不覆盖该领域。请使用自身知识或联网搜索结果来回答。

规则：
1. 在回答开头简要声明："项目文档库中未找到相关内容"（一句即可，不要反复强调）
2. 然后正常回答问题——基于联网搜索结果优先，其次基于自身知识
3. 文档未覆盖的陈述标注 [通用知识]，联网搜索的内容标注 [网络搜索]
4. 如果你也不确定答案，直接说"不确定"，不要编造
5. 不要因为文档中没有就拒绝回答——用户问的是问题，不是文档查询

示例输出：
"文档库中未找到与Linux内核架构相关的内容，以下基于通用知识：

Linux是一个开源的操作系统内核 [通用知识]。它由Linus Torvalds于1991年创建 [通用知识]。内核负责进程调度、内存管理和设备驱动 [通用知识]。

如果您需要基于项目文档的准确回答，建议上传相关文档后再次询问。"
"""


def build_degradation_answer(
    llm_client,
    user_query: str,
    max_level: int = 1,  # CHITCHAT=0, SHALLOW=2, DEEP=2
) -> DegradationResult:
    """构建降级链回答。

    Level 0: 诚实声明（总是执行）
    Level 1: LLM 自身知识回答（总是执行）
    Level 2: MCP 联网搜索（仅 max_level >= 2 时执行）

    返回 DegradationResult。
    """
    declaration = "项目文档库中未找到与您问题相关的内容。"

    result = DegradationResult(
        level0_declaration=declaration,
        used_levels=[0],
    )

    # Level 1: LLM 自身知识
    try:
        level1_answer = _generate_level1_answer(llm_client, user_query, declaration)
        result.level1_answer = level1_answer
        result.used_levels.append(1)
        result.sources_labeled = True
    except Exception:
        logger.exception("Degradation Level 1 failed")
        result.level1_answer = f"{declaration}\n\n很抱歉，当前无法生成回答。请稍后重试或提供相关文档。"
        result.sources_labeled = True

    # Level 2: MCP 联网搜索 + LLM 整合（仅 deep）
    if max_level >= 2:
        try:
            web_results = _try_web_search(llm_client, user_query)
            if web_results:
                result.level2_web_results = web_results
                result.used_levels.append(2)
                # 让 LLM 整合网络搜索结果，生成带 [网络搜索] 标注的回答
                result.level2_integrated_answer = _generate_level2_answer(
                    llm_client, user_query, declaration, web_results,
                )
        except Exception:
            logger.exception("Degradation Level 2 (web search) failed")

    logger.info(
        "Degradation chain | q=%.50s levels=%s max=%d",
        user_query, result.used_levels, max_level,
    )
    return result


def _generate_level1_answer(
    llm_client,
    user_query: str,
    declaration: str,
) -> str:
    """生成 Level 1 回答：LLM 自身知识 + [通用知识] 标注。"""
    messages = [
        {"role": "system", "content": DEGRADATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    answer = llm_client.chat(messages, temperature=0.5)
    return answer


def _generate_level2_answer(
    llm_client,
    user_query: str,
    declaration: str,
    web_results: list[dict],
) -> str:
    """生成 Level 2 回答：LLM 整合联网搜索结果 + [网络搜索] 标注。"""
    # 格式化网络搜索结果
    web_parts = []
    for i, wr in enumerate(web_results, 1):
        title = wr.get("title", "")
        snippet = wr.get("snippet", "")
        url = wr.get("url", "")
        web_parts.append(f"[结果 {i}] {title}\n{snippet}\n来源: {url}")
    web_context = "\n\n".join(web_parts)

    system_prompt = (
        "你是 Mini Agent，一个IT运维知识助手。\n\n"
        f"{declaration}\n"
        "以下信息来自**联网搜索**，请基于这些信息回答用户问题。\n\n"
        "规则：\n"
        "1. 优先使用联网搜索结果中的信息\n"
        "2. 所有来自网络搜索的内容标注 [网络搜索]\n"
        "3. 基于自身知识补充或推断的内容标注 [通用知识]\n"
        "4. 如果搜索结果也不足以全面回答，诚实说明\n"
        "5. 不要在正文末尾列出参考来源、URL 清单或来源编号，系统会在来源追踪区展示网页来源\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"联网搜索结果：\n\n{web_context}"},
        {"role": "user", "content": user_query},
    ]

    try:
        answer = llm_client.chat(messages, temperature=0.5)
    except Exception:
        logger.exception("Level 2 answer generation failed")
        # 回退到 Level 1
        return _generate_level1_answer(llm_client, user_query, declaration)

    return answer


def _try_web_search(
    llm_client,
    user_query: str,
) -> list[dict]:
    """尝试联网搜索。使用配置的搜索后端（DuckDuckGo / Brave Search）。"""
    try:
        from app.services.web_search_service import search_web
        results = search_web(user_query)
        if results:
            logger.info("Web search | q=%.50s → %d results", user_query, len(results))
        else:
            logger.info("Web search | q=%.50s → no results (disabled or failed)", user_query)
        return results
    except Exception:
        logger.exception("Web search failed")
        return []


def format_degradation_response(result: DegradationResult) -> str:
    """将降级链结果格式化为最终回答字符串。

    Level 2 优先：如果 LLM 已整合网络搜索结果，直接返回整合后的回答。
    否则返回 Level 0 声明 + Level 1 通用知识回答。
    """
    # Level 2: LLM 已整合网络搜索结果 → 直接返回
    if 2 in result.used_levels and result.level2_integrated_answer:
        return result.level2_integrated_answer

    # Level 0 + Level 1: 诚实声明 + 通用知识
    parts = [result.level0_declaration]
    if 1 in result.used_levels and result.level1_answer:
        parts.append("")
        parts.append(result.level1_answer)

    return "\n".join(parts)


DEGRADATION_FALLBACK_SYSTEM_PROMPT = """你是 Mini Agent。项目文档库检索到一些内容但质量不足以充分回答问题。

你必须：
1. 在回答中先说明检索情况：哪些部分有文档支撑，哪些没有
2. 有文档引用的部分正常回答，标注 [项目文档]
3. 无文档支撑的部分标注 [通用知识]
4. 在回答末尾列出"不确定项"清单

不要编造文档中不存在的信息。"""
