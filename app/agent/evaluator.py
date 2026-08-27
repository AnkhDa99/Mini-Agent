"""EvaluatorAgent — 独立的质量评判 Agent。

使用轻量模型（deepseek-v4-flash）专做检索质量评判：
- 完整性（completeness）：检索结果能否充分回答用户问题
- 有据性（groundedness）：回答有多少文档引用支撑
- 裁决（decision）：CONTINUE / STOP

独立于生成模型的"第二双眼睛"，消除"自己评自己"的偏差。
"""

from __future__ import annotations

import logging

from app.agent.quality_signal import (
    COMPLETENESS_MINIMUM,
    CONCENTRATION_GOOD,
    QualitySignal,
)
from app.llm.model_registry import get_model_registry

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """你是 Agent 质量评判器（Evaluator）。基于检索结果评估信息充分性。

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
   - CONTINUE: 信息不充分，需要继续检索
   - STOP: 信息已经足够回答用户问题

5. **uncertainty_handling**：
   - retry: 换策略再检索一轮
   - admit: 诚实承认文档库未覆盖
   - web_search: 建议联网搜索
   - general_knowledge: 用通用知识回答

输出格式（严格 JSON）：
{"completeness": 75, "groundedness": 60, "uncertainties": ["性能数据缺失"], "decision": "CONTINUE", "decision_reason": "缺少性能对比数据", "uncertainty_handling": "retry"}"""

QUICK_CHECK_PROMPT = """判断以下检索结果能否回答用户问题。
只输出一个 0-100 的整数：
90-100: 完整覆盖
70-89: 核心覆盖，缺少细节
40-69: 部分覆盖，关键信息缺失
0-39: 基本不相关，无法回答
只输出数字。"""


class EvaluatorAgent:
    """独立评判 Agent，使用轻量模型评估检索质量。"""

    def __init__(self, overrides: dict | None = None):
        self._client = get_model_registry().get_client("evaluator", overrides or {})
        logger.info("EvaluatorAgent init | model=%s", self._client.model)

    @property
    def llm(self):
        return self._client

    # ── DEEP 完整评估 ──

    def assess(
        self,
        user_query: str,
        search_results: list[dict],
        quality_signal: QualitySignal,
        existing_uncertainties: list[str] | None = None,
    ) -> QualitySignal:
        """完整 Critic 评估：concentration 信号 + LLM 自评。

        返回更新后的 QualitySignal（含 completeness/groundedness/decision）。
        """
        # 空结果 → 不调 LLM，直接判定
        if quality_signal.is_empty:
            logger.info("Evaluator: empty results, no LLM assessment needed")
            return quality_signal.merge_llm_assessment(
                completeness=0,
                groundedness=0,
                uncertainties=existing_uncertainties or ["文档库未覆盖此问题"],
                decision="STOP",
                decision_reason="检索完全无结果，触发降级",
                uncertainty_handling="admit",
            )

        # 构建评估上下文
        context_summary = self._build_context(search_results, quality_signal)
        uncertainties_text = ""
        if existing_uncertainties:
            uncertainties_text = f"已标记的不确定项：{', '.join(existing_uncertainties)}"

        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_query}\n\n"
                    f"检索信号：concentration={quality_signal.concentration:.2f}, "
                    f"info_gain={quality_signal.info_gain:.2f}\n"
                    f"命中 chunk 数：{quality_signal.results_count}\n\n"
                    f"检索结果摘要：\n{context_summary}\n\n"
                    f"{uncertainties_text}"
                ),
            },
        ]

        try:
            result = self.llm.chat_json(messages, temperature=0.2)
            if not result:
                raise ValueError("Evaluator LLM returned empty")

            quality_signal.merge_llm_assessment(
                completeness=int(result.get("completeness", 50)),
                groundedness=int(result.get("groundedness", 50)),
                uncertainties=result.get("uncertainties", []),
                decision=result.get("decision", "CONTINUE"),
                decision_reason=result.get("decision_reason", ""),
                uncertainty_handling=result.get("uncertainty_handling", "retry"),
            )
        except Exception:
            logger.exception("Evaluator LLM assessment failed, using signal fallback")
            if quality_signal.is_good and quality_signal.concentration >= CONCENTRATION_GOOD:
                decision, reason = "STOP", "signal: good quality + high concentration (fallback)"
            else:
                decision, reason = "CONTINUE", "signal: insufficient quality (fallback)"

            quality_signal.merge_llm_assessment(
                completeness=60 if quality_signal.is_good else 30,
                groundedness=60 if quality_signal.is_good else 30,
                decision=decision,
                decision_reason=reason,
                uncertainty_handling="retry",
            )

        logger.info(
            "Evaluator | grade=%s conc=%.2f gain=%.2f comp=%d gnd=%d → %s",
            quality_signal.retrieval_grade,
            quality_signal.concentration,
            quality_signal.info_gain,
            quality_signal.completeness,
            quality_signal.groundedness,
            quality_signal.decision,
        )
        return quality_signal

    # ── SHALLOW 快速完整性检查 ──

    def quick_completeness_check(
        self,
        user_query: str,
        search_results: list[dict],
    ) -> int:
        """SHALLOW / 文档生成后的快速完整性检查。

        只判断 completeness 0-100，不做完整 Critic 评估。
        用于兜底：浓度过关但内容不相关时不直接生成低质量回答。
        """
        if not search_results:
            return 0

        snippets = []
        for i, r in enumerate(search_results[:5], 1):
            content = (r.get("content") or "")[:150]
            snippets.append(f"[{i}] {content}...")
        context = "\n".join(snippets) if snippets else "（无有效内容）"

        messages = [
            {"role": "system", "content": QUICK_CHECK_PROMPT},
            {"role": "user", "content": f"用户问题：{user_query}\n\n检索结果：\n{context}"},
        ]

        try:
            response = self.llm.chat(messages, temperature=0.1)
            import re
            match = re.search(r"\d+", response)
            if match:
                return min(max(int(match.group()), 0), 100)
            return 50
        except Exception:
            logger.exception("Quick completeness check failed")
            return 50

    # ── 辅助方法 ──

    @staticmethod
    def _build_context(
        results: list[dict],
        quality_signal: QualitySignal,
    ) -> str:
        """构建评估用的检索结果摘要。"""
        if not results:
            return "（无检索结果）"

        lines = []
        for i, r in enumerate(results[:5], 1):
            content = (r.get("content") or "")[:200]
            filename = r.get("filename", "未知文档")
            score = r.get("rrf_score", r.get("faiss_score", 0))
            lines.append(f"[{i}] {filename} (分数={score:.3f}): {content}...")

        lines.append(
            f"\n信号: concentration={quality_signal.concentration:.2f}, "
            f"info_gain={quality_signal.info_gain:.2f}"
        )
        return "\n".join(lines)


# 全局单例
_evaluator: EvaluatorAgent | None = None


def get_evaluator(overrides: dict | None = None) -> EvaluatorAgent:
    global _evaluator
    if _evaluator is None or overrides:
        _evaluator = EvaluatorAgent(overrides=overrides)
    return _evaluator
