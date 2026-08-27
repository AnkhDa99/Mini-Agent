"""
Critic — 检索质量 + 回答充分性自检。

每轮 Agent Think 后由 Critic 评估：
1. 检索质量：基于信号（cosine、concentration、info_gain）
2. 回答充分性：LLM 自评（completeness、groundedness）

输出结构化终止信号：CONTINUE / STOP
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CriticAssessment:
    """Critic 评估结果。"""
    # 检索质量（信号驱动）
    retrieval_quality: str  # "good" | "marginal" | "poor" | "empty"
    faiss_ok: bool
    concentration: float
    info_gain: float = 0.0

    # 回答充分性（LLM 自评）
    completeness: int = 0     # 0-100
    groundedness: int = 0     # 0-100
    uncertainties: list[str] = field(default_factory=list)
    decision: str = "CONTINUE"  # CONTINUE | STOP
    decision_reason: str = ""
    uncertainty_handling: str = "retry"  # retry | admit | web_search | general_knowledge


CRITIC_SYSTEM_PROMPT = """你是 Agent 自检器（Critic）。基于当前检索结果和已收集的信息，评估回答的充分性。

评估维度：
1. **completeness（0-100）**：当前收集的信息能多大程度上完整回答用户问题？
   - 90-100: 核心内容和细节都已覆盖
   - 70-89: 核心内容已覆盖，缺少一些细节
   - 40-69: 只覆盖了一部分，关键信息缺失
   - 0-39: 基本没有有效信息

2. **groundedness（0-100）**：有多少结论有文档引用支撑？
   - 90-100: 几乎全部有引用
   - 70-89: 大部分有引用，少数来自通用知识
   - 40-69: 只有部分有引用
   - 0-39: 基本没有文档支撑

3. **uncertainties**：列出当前仍不确定的具体项目

4. **decision**：
   - CONTINUE: 信息不充分，需要继续检索或搜索
   - STOP: 信息已经足够回答用户问题

5. **uncertainty_handling**：
   - retry: 换策略再检索一轮
   - admit: 诚实承认文档库未覆盖，标注 [通用知识]
   - web_search: 建议联网搜索（仅 deep 问题可用）
   - general_knowledge: 用通用知识回答+标注

输出格式（严格 JSON，不要其他内容）：
{"completeness": 75, "groundedness": 60, "uncertainties": ["性能数据缺失"], "decision": "CONTINUE", "decision_reason": "缺少性能对比数据", "uncertainty_handling": "retry"}"""


def assess_retrieval_quality(fusion_info: dict | None) -> tuple[str, bool, float]:
    """基于信号评估检索质量（无 LLM 调用）。

    返回: (quality, faiss_ok, concentration)
    """
    if not fusion_info:
        return ("empty", False, 0.0)

    faiss_ok = fusion_info.get("faiss_ok", False)
    es_ok = fusion_info.get("es_ok", False)
    faiss_conc = fusion_info.get("faiss_conc", 0)
    es_conc = fusion_info.get("es_conc", 0)
    concentration = max(faiss_conc, es_conc)

    if not faiss_ok and not es_ok:
        return ("empty", False, 0.0)

    if concentration >= 0.5:
        return ("good", True, concentration)
    elif concentration >= 0.3:
        return ("marginal", faiss_ok, concentration)
    else:
        return ("poor", faiss_ok, concentration)


def compute_info_gain(
    new_results: list[dict],
    seen_chunk_uids: set[str],
) -> float:
    """计算新检索结果的信息增益。

    增益 = 新 unique chunk 数 / 新结果总数
    范围 [0, 1]。
    """
    if not new_results:
        return 0.0
    new_uids = {r.get("chunk_uid", "") for r in new_results}
    novel = new_uids - seen_chunk_uids
    return len(novel) / len(new_results)


def critic_assess(
    llm_client,
    user_query: str,
    search_results: list[dict],
    fusion_info: dict | None,
    info_gain: float,
    existing_uncertainties: list[str] | None = None,
) -> CriticAssessment:
    """Critic 主入口：信号分析 + LLM 自评。

    返回 CriticAssessment。
    """
    # 1. 检索质量评估（纯信号，不调 LLM）
    quality, faiss_ok, concentration = assess_retrieval_quality(fusion_info)

    if quality == "empty":
        # 完全无结果 → 不需要 LLM 自评，直接判定
        logger.info("Critic: retrieval empty, no LLM assessment needed")
        return CriticAssessment(
            retrieval_quality="empty",
            faiss_ok=False,
            concentration=0.0,
            info_gain=0.0,
            completeness=0,
            groundedness=0,
            uncertainties=existing_uncertainties or ["文档库未覆盖此问题"],
            decision="STOP",
            decision_reason="检索完全无结果，触发降级链",
            uncertainty_handling="admit",
        )

    # 2. LLM 自评
    # 构建检索上下文摘要给 Critic
    context_summary = _build_critic_context(search_results, fusion_info, info_gain)

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{user_query}\n\n"
                f"检索信号：cosine_ok={faiss_ok}, concentration={concentration:.2f}, info_gain={info_gain:.2f}\n"
                f"命中 chunk 数：{len(search_results)}\n\n"
                f"检索结果摘要：\n{context_summary}\n\n"
                f"{'已标记的不确定项：' + ', '.join(existing_uncertainties) if existing_uncertainties else ''}"
            ),
        },
    ]

    try:
        result = llm_client.chat_json(messages, temperature=0.2)
        if not result:
            raise ValueError("Critic LLM returned empty")

        assessment = CriticAssessment(
            retrieval_quality=quality,
            faiss_ok=faiss_ok,
            concentration=concentration,
            info_gain=info_gain,
            completeness=int(result.get("completeness", 50)),
            groundedness=int(result.get("groundedness", 50)),
            uncertainties=result.get("uncertainties", []),
            decision=result.get("decision", "CONTINUE"),
            decision_reason=result.get("decision_reason", ""),
            uncertainty_handling=result.get("uncertainty_handling", "retry"),
        )
    except Exception:
        logger.exception("Critic LLM assessment failed")
        # 兜底：信号驱动判断
        if quality == "good" and concentration >= 0.5:
            decision = "STOP"
            reason = "signal: good quality + high concentration (fallback)"
        else:
            decision = "CONTINUE"
            reason = "signal: insufficient quality (fallback)"

        assessment = CriticAssessment(
            retrieval_quality=quality,
            faiss_ok=faiss_ok,
            concentration=concentration,
            info_gain=info_gain,
            completeness=60 if quality == "good" else 30,
            groundedness=60 if quality == "good" else 30,
            decision=decision,
            decision_reason=reason,
            uncertainty_handling="retry",
        )

    logger.info(
        "Critic | q=%.40s quality=%s conc=%.2f gain=%.2f comp=%d gnd=%d → %s",
        user_query, quality, concentration, info_gain,
        assessment.completeness, assessment.groundedness, assessment.decision,
    )

    return assessment


def _build_critic_context(
    results: list[dict],
    fusion_info: dict | None,
    info_gain: float,
) -> str:
    """构建 Critic 评估用的上下文摘要。"""
    if not results:
        return "（无检索结果）"

    lines = []
    # 取前 5 个结果做摘要
    for i, r in enumerate(results[:5], 1):
        content = (r.get("content") or "")[:200]
        filename = r.get("filename", "未知文档")
        score = r.get("rrf_score", r.get("faiss_score", 0))
        lines.append(f"[{i}] {filename} (分数={score:.3f}): {content}...")

    if fusion_info:
        lines.append(
            f"\n信号: faiss_ok={fusion_info.get('faiss_ok')}, "
            f"es_ok={fusion_info.get('es_ok')}, "
            f"concentration={fusion_info.get('faiss_conc', fusion_info.get('es_conc', 0)):.2f}, "
            f"info_gain={info_gain:.2f}"
        )

    return "\n".join(lines)
