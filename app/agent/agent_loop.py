"""
Agent 循环：Think → Act → Observe → Critic → 终止判断。

支持两种模式：
1. Function Calling 模式（首选）：LLM 原生支持 tools 参数
2. Prompt 模式（兜底）：LLM 输出 JSON，由框架解析执行

终止条件（四层，按优先级）：
  L1: round >= max_rounds（硬上限）
  L2: 连续 2 轮未调检索工具（工具模式检测）
  L3: 连续 2 轮 info_gain < 20%（信息增益衰减）
  L4: 连续 2 轮 Critic STOP（模型自判）

降级链（文档库无结果时触发）：
  Level 0: 诚实声明
  Level 1: LLM 自身知识 + [通用知识] 标注
  Level 2: MCP 联网搜索 + [网络搜索] 标注（SHALLOW/DEEP 均可触发）
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent.classifier import ClassificationResult, QueryClass, classify
from app.agent.critic import (
    CriticAssessment,
    compute_info_gain,
)
from app.agent.degradation import (
    DegradationResult,
    build_degradation_answer,
    format_degradation_response,
)
from app.agent.quality_signal import (
    COMPLETENESS_MINIMUM,
    CONCENTRATION_MARGINAL,
    INFO_GAIN_MINIMUM,
    QualitySignal,
)
from app.agent.evaluator import get_evaluator
from app.agent.tool_registry import (
    get_tool_specs_for_class,
    get_tool_category,
)
from app.llm.model_registry import get_model_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent 循环状态。"""
    user_query: str
    classification: ClassificationResult

    # 轮次
    round: int = 0
    max_rounds: int = 1

    # 检索状态
    search_results: list[dict] = field(default_factory=list)
    seen_chunk_uids: set[str] = field(default_factory=set)
    fusion_info: dict | None = None
    last_info_gain: float = 0.0
    all_results: list[dict] = field(default_factory=list)  # 累积所有轮次的结果

    # 终止信号计数
    consecutive_non_search_rounds: int = 0
    consecutive_low_gain_rounds: int = 0
    consecutive_stop_decisions: int = 0

    # 工具调用历史
    tool_call_history: list[dict] = field(default_factory=list)

    # Critic 历史
    critic_history: list[CriticAssessment] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    # 降级
    degradation_triggered: bool = False
    degradation_result: DegradationResult | None = None

    # 统一质量信号（Evaluator Agent 填充）
    quality_signal: Any | None = None

    # 是否已执行文档生成
    doc_gen_done: bool = False

    # 前端强制联网搜索
    force_web_search: bool = False
    allow_web_search: bool = True

    # 是否应抑制来源展示（LLM 判断检索结果与问题无关）
    suppress_sources: bool = False

    # 场景知识库匹配结果
    scenario_matches: list[dict] = field(default_factory=list)
    scenario_match_score: float = 0.0

    # 是否已生成最终答案
    final_answer: str = ""

    # 耗时
    timings: dict[str, float] = field(default_factory=dict)


AGENT_SYSTEM_PROMPT = """你是 Mini Agent，一个企业项目知识引擎。你可以使用工具来检索项目文档、分析信息并生成回答。

## 工作流程
1. 分析用户问题，决定需要调用什么工具
2. 调用工具获取信息
3. 评估获取的信息是否足够
4. 如果不够，考虑更换策略继续检索
5. 信息充分后，调用 generate_answer 输出最终答案

## 核心原则
- **必须检索**：所有非闲聊问题必须先检索项目文档库
- **诚实**：文档覆盖不到的内容，标注 [通用知识]；联网搜索的内容标注 [网络搜索]
- **高效**：不要重复无效检索。如果连续 2 轮都没新信息，就应该停止检索并诚实回答
- **来源标注**：最终回答中，有文档引用支撑的结论不标注，来自自身知识的标注 [通用知识]；场景知识库的内容标注 [知识库]
- **话题切换意识**：用户可能在上一条消息中讨论话题A，下一条切换到话题B。你要独立分析当前问题，不要因为上一轮检索的文档与当前问题无关就拒绝回答。如果文档库不覆盖当前话题，使用 web_search 或自身知识回答
- **不拒绝原则**：即使用户问题与已上传文档完全无关（例如用户上传了公司A的文档，但问的是公司B），也要尽力回答。优先 web_search，其次使用自身通用知识，不要直接说"文档中未找到"就结束

## 检索策略
- search_knowledge_base: 基础检索（Concentration-RRF），总是先用这个
- search_scenario_kb: 运维场景知识库检索。当用户询问故障排查、报错处理、性能问题等运维场景时，**强烈建议在 search_knowledge_base 之后调用此工具**。场景知识库包含 MySQL/Redis/K8s/Nginx/Kafka/ES/Docker/Java 等常见故障的结构化排障知识
- rewrite_query: 改写查询措辞以提升召回（用于 cosine < 0.7 时）。如果查询含中文实体名，尝试生成英文/拼音变体
- web_search: 联网搜索（当文档库检索结果为空或与问题不相关时使用）
- multi_query_search: 生成多个视角的查询变体（用于问题涉及多个子话题）
- hyde_search: 生成假设文档片段再检索（用于用词不匹配时）
- rerank_results: 精排已有结果（用于 concentration 低时）
- decompose_task: 拆解复杂问题为子问题分别检索

## 文档生成（在检索完成后调用）
如果用户要求「写文档」「生成报告」「制作PPT」「画图」「生成表格」等，检索完成后不要直接 generate_answer，而是调用以下工具生成文档，再把文档信息传给 generate_answer：
- generate_word: 生成 Word 文档（技术方案/需求文档/项目总结/会议纪要）
- generate_ppt: 生成 PPT 演示文稿
- generate_pdf: 生成 PDF 文档
- generate_excel: 生成 Excel 表格（数据汇总/对比分析/排期计划/清单列表）
- generate_diagram: 生成架构图/流程图/时序图/类图（Mermaid 格式）
- generate_markdown: 生成 Markdown 文档（.md 文件）

工作流程变为：检索 → 确认信息充分 → 调用文档生成工具 → generate_answer 引用生成的文档

## 终止条件
当你认为信息已经足够回答用户问题时，调用 generate_answer 输出最终答案。
如果你发现连续检索没有新信息，应该停止并诚实告知用户。

## 当前信号
系统会在每轮给你当前的检索信号（cosine、concentration、info_gain），
请根据信号值决定下一步策略。"""


def _detect_doc_gen_intent(
    query: str,
    classification: ClassificationResult,
) -> tuple[str, dict] | None:
    """检测用户是否需要生成文档，返回 (tool_name, tool_args) 或 None。

    仅在 DEEP 分类下检测，避免浅层问题触发文档生成。
    """
    if classification.query_class == QueryClass.CHITCHAT:
        return None

    q_lower = query.lower()

    # 图表
    for kw in ["架构图", "流程图", "时序图", "类图"]:
        if kw in query:
            return ("generate_diagram", {
                "diagram_type": kw,
                "description": query,
            })
    if any(kw in q_lower for kw in ["画图", "图表", "mermaid", "关系图", "flowchart"]):
        return ("generate_diagram", {
            "diagram_type": "architecture",
            "description": query,
        })

    # Markdown
    if any(kw in q_lower for kw in ["md", "markdown", ".md"]):
        return ("generate_markdown", {"title": ""})

    # PPT
    if any(kw in q_lower for kw in ["ppt", "演示", "幻灯片", "答辩", "汇报"]):
        return ("generate_ppt", {"title": "", "outline": ""})

    # Excel
    if any(kw in q_lower for kw in ["表格", "excel", "清单", "排期", "对比"]):
        sheet_type = "数据汇总"
        if "排期" in query:
            sheet_type = "排期计划"
        elif "对比" in query:
            sheet_type = "对比分析"
        elif "清单" in query:
            sheet_type = "清单列表"
        return ("generate_excel", {"title": "", "sheet_type": sheet_type})

    # PDF
    if any(kw in q_lower for kw in ["pdf", "归档"]):
        return ("generate_pdf", {"title": ""})

    # Word（最高频：写文档/生成文档/做报告 等）—— 放在最后作为兜底
    # 要求明确的文档生成意图，避免 "写代码/生成UUID" 等误触发
    doc_kw = [
        "写文档", "写报告", "写方案", "写总结", "写纪要",
        "生成文档", "生成报告", "制作文档", "创建文档",
        "输出文档", "输出报告", "做文档", "做报告",
        "周报", "日报", "纪要",
    ]
    if any(kw in query for kw in doc_kw):
        content_type = "技术方案"
        if "周报" in query:
            content_type = "项目周报"
        elif "日报" in query:
            content_type = "项目日报"
        elif "总结" in query:
            content_type = "项目总结"
        elif "纪要" in query:
            content_type = "会议纪要"
        return ("generate_word", {"title": "", "content_type": content_type})

    return None


def _check_termination(state: AgentState) -> tuple[bool, str]:
    """检查四层终止条件。

    返回 (should_terminate, reason)。
    按优先级 L1 → L2 → L3 → L4 依次检查。
    """
    # L1: 硬上限（> 非 >=，while 循环边界由 round < max_rounds 控制）
    if state.round > state.max_rounds:
        return True, f"L1: 超过最大轮次上限 ({state.round} > {state.max_rounds})"

    # L2: 工具调用模式 — 连续 2 轮未调用检索类工具
    if state.consecutive_non_search_rounds >= 2:
        return True, "L2: 连续2轮未调用检索工具，Agent 正在生成答案"

    # L3: 信息增益衰减
    if state.consecutive_low_gain_rounds >= 2:
        return True, "L3: 连续2轮信息增益不足20%"

    # L4: 模型自判
    if state.consecutive_stop_decisions >= 2:
        return True, "L4: 连续2轮模型自判信息充分"

    return False, ""


def _update_termination_counters(state: AgentState, last_critic: CriticAssessment | None):
    """更新终止计数器。"""
    # L2: 检查上一轮是否调用了检索工具
    if state.tool_call_history:
        last_round_tools = [
            tc for tc in state.tool_call_history
            if tc.get("round") == state.round
        ]
        search_tools_used = any(
            get_tool_category(tc.get("name", "")) == "search"
            for tc in last_round_tools
        )
        if not search_tools_used:
            state.consecutive_non_search_rounds += 1
        else:
            state.consecutive_non_search_rounds = 0

    # L3: info_gain
    if state.last_info_gain < INFO_GAIN_MINIMUM:
        state.consecutive_low_gain_rounds += 1
    else:
        state.consecutive_low_gain_rounds = 0

    # L4: Critic STOP
    if last_critic and last_critic.decision == "STOP":
        state.consecutive_stop_decisions += 1
    else:
        state.consecutive_stop_decisions = 0


def _update_termination_counters_from_signal(state: AgentState, signal: "QualitySignal"):
    """从 QualitySignal 更新终止计数器（替代 CriticAssessment 版本）。"""
    # L2: 检查上一轮是否调用了检索工具
    if state.tool_call_history:
        last_round_tools = [
            tc for tc in state.tool_call_history
            if tc.get("round") == state.round
        ]
        search_tools_used = any(
            get_tool_category(tc.get("name", "")) == "search"
            for tc in last_round_tools
        )
        if not search_tools_used:
            state.consecutive_non_search_rounds += 1
        else:
            state.consecutive_non_search_rounds = 0

    # L3: info_gain
    if state.last_info_gain < INFO_GAIN_MINIMUM:
        state.consecutive_low_gain_rounds += 1
    else:
        state.consecutive_low_gain_rounds = 0

    # L4: Evaluator STOP
    if signal.decision == "STOP":
        state.consecutive_stop_decisions += 1
    else:
        state.consecutive_stop_decisions = 0


def _build_agent_messages(state: AgentState, initial_search_done: bool) -> list[dict]:
    """构建 Agent think 阶段的 messages。"""
    from app.core.config import settings
    system_prompt = AGENT_SYSTEM_PROMPT
    if not settings.scenario_kb_enabled:
        # 移除场景知识库相关行
        system_prompt = "\n".join(
            line for line in system_prompt.split("\n")
            if "search_scenario_kb" not in line and "场景知识库" not in line
        )
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # 当前状态
    status_parts = [
        f"用户问题：{state.user_query}",
        f"问题类型：{state.classification.query_class.value}",
        f"当前轮次：{state.round}/{state.max_rounds}",
        f"最大可用轮次：{state.max_rounds}",
    ]

    if state.fusion_info:
        fi = state.fusion_info
        status_parts.append(
            f"检索信号：faiss_ok={fi.get('faiss_ok')}, "
            f"es_ok={fi.get('es_ok')}, "
            f"concentration={fi.get('faiss_conc', fi.get('es_conc', 0)):.2f}, "
            f"info_gain={state.last_info_gain:.2f}"
        )

    if state.all_results:
        status_parts.append(f"已累计检索结果：{len(state.all_results)} 条")
    if state.scenario_matches:
        sc_titles = [m.get('title', '')[:20] for m in state.scenario_matches[:3]]
        status_parts.append(f"已匹配场景知识库：{len(state.scenario_matches)} 条 ({', '.join(sc_titles)})")
    if state.uncertainties:
        status_parts.append(f"当前不确定项：{', '.join(state.uncertainties)}")

    messages.append({"role": "user", "content": "\n".join(status_parts)})

    # 工具调用历史
    if state.tool_call_history:
        history_text = "工具调用历史：\n"
        for tc in state.tool_call_history:
            result_summary = str(tc.get("result_summary", ""))[:200]
            history_text += (
                f"- Round {tc.get('round')}: {tc.get('name')} "
                f"→ {result_summary}\n"
            )
        messages.append({"role": "user", "content": history_text})

    # 指令
    if not initial_search_done:
        if state.force_web_search:
            instruction = "用户已开启联网搜索。请先调用 search_knowledge_base 检索项目文档。如果无结果，立即调用 web_search。"
        else:
            instruction = "请先调用 search_knowledge_base 检索项目文档。"
    elif state.force_web_search:
        instruction = (
            "用户要求联网搜索！\n"
            "- 如果文档检索结果足以回答问题 → 直接调用 generate_answer\n"
            "- 如果文档检索结果为空或不相关 → **必须调用 web_search 进行联网搜索**，不要跳过\n"
            "- 不要直接说文档中找不到就结束——用户期望你使用搜索引擎"
        )
    elif state.classification.query_class == QueryClass.SHALLOW:
        instruction = (
            "你还有 1 轮检索机会。根据检索信号决定：\n"
            "- 信号好（concentration > 0.5）→ 直接调用 generate_answer\n"
            "- 信号一般 → 调用 rewrite_query 尝试改善（如果查询含中文，生成英文变体）\n"
        )
        if settings.scenario_kb_enabled:
            instruction += "- 如果问题涉及故障排查 → 务必调用 search_scenario_kb 检索运维排障知识\n"
        if state.allow_web_search:
            instruction += (
                "- 无文档命中或检索结果与当前问题明显不相关 → 调用 web_search 联网搜索\n"
                "- 不要因为文档中找不到就说'不知道'——先用 web_search，再用自身知识"
            )
        else:
            instruction += "- 当前用户未启用联网搜索；无文档命中时请使用通用知识补充，并明确标注 [通用知识]"
    else:
        # 检测用户是否需要文档生成
        doc_keywords = ["写", "生成", "制作", "画", "创建", "输出", "导出",
                        "文档", "报告", "ppt", "word", "pdf", "excel", "表格",
                        "图表", "架构图", "流程图", "时序图", "类图", "方案",
                        "周报", "日报", "总结", "纪要", "清单", "排期"]
        q_lower = state.user_query.lower()
        want_doc = any(kw in q_lower for kw in doc_keywords)

        if want_doc:
            if state.doc_gen_done:
                # 文档已生成，直接回答
                instruction = (
                    "文档已生成完毕。现在调用 generate_answer 输出回答，"
                    "必须在回答中告知用户文档已生成并提供下载链接。"
                )
            else:
                instruction = (
                    "用户需要生成文档！检索已完成，现在必须调用文档生成工具。\n"
                    "判断用户需求并选择：\n"
                    "- 写文档/报告/方案 → generate_word\n"
                    "- 做PPT/演示 → generate_ppt\n"
                    "- 生成PDF → generate_pdf\n"
                    "- 表格/对比/清单 → generate_excel\n"
                    "- 架构图/流程图 → generate_diagram\n"
                    "- Markdown文档 → generate_markdown\n\n"
                    "调用生成工具后，再用 generate_answer 引用生成的文档链接。"
                    "不要跳过生成步骤直接回答！"
                )
        else:
            instruction = (
                "根据检索信号决定下一步：\n"
                "- 信号好 → generate_answer\n"
                "- concentration低 → rerank_results 或 multi_query_search\n"
                "- cosine低 → rewrite_query（含中文实体名时生成英文变体）或 hyde_search\n"
            )
            if settings.scenario_kb_enabled:
                instruction += "- 如果问题涉及故障排查/报错/性能问题 → **务必调用 search_scenario_kb**\n"
            instruction += "- 信息不足 → 换个策略再查\n"
            if state.allow_web_search:
                instruction += (
                    "- 无文档命中或检索结果与当前问题明显不相关 → **先调用 web_search**，"
                    "不要直接 say 不知道。用户可能切换了话题，文档库不覆盖是正常的，"
                    "用联网搜索或自身知识回答即可"
                )
            else:
                instruction += "- 当前用户未启用联网搜索；文档库不覆盖时请使用自身通用知识回答，并标注 [通用知识]"

    messages.append({"role": "user", "content": instruction})

    return messages


def _emit(progress_callback, event: dict):
    """安全调用进度回调。"""
    if progress_callback:
        try:
            progress_callback(event)
        except Exception:
            pass  # 回调失败不影响主流程


def run_agent_loop(
    user_query: str,
    search_fn: Callable[[str], tuple[list[dict], dict]],
    llm_client=None,
    stream: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    force_web_search: bool = False,
    allow_web_search: bool = True,
    model_overrides: dict | None = None,
) -> AgentState:
    """Agent 循环主入口（多 Agent 编排）。

    内部使用 ModelRegistry 自动分配合适的模型：
      - Classifier Agent: classifier_model (flash)
      - Evaluator Agent:  evaluator_model  (flash)
      - Generator Agent:  generator_model  (pro)
      - Doc Generator:    doc_generator_model (flash)

    Args:
        user_query: 用户原始问题
        llm_client: (可选，向后兼容) 旧版单一 LLM 客户端。新架构从 ModelRegistry 获取。
        search_fn: 检索函数 (query) → (results, fusion_info)
        stream: 是否流式输出
        progress_callback: 进度回调，每阶段推送 event dict 给前端
        force_web_search: 前端强制开启联网搜索
        allow_web_search: 当前用户/环境是否允许使用联网搜索
        model_overrides: 前端传来的运行时模型配置（Issue 2）

    Returns:
        AgentState: 包含最终回答和完整状态
    """
    t0 = time.perf_counter()
    force_web_search = bool(force_web_search and allow_web_search)
    state = AgentState(user_query=user_query, force_web_search=force_web_search, allow_web_search=allow_web_search, classification=ClassificationResult(
        query_class=QueryClass.SHALLOW, confidence=0.3,
        reason="initial", max_rounds=2, max_tool_calls_per_round=1,
        allowed_tool_categories=["search"], degradation_max_level=1,
    ))

    # ── 多模型注册（支持运行时覆盖，Issue 2）──
    registry = get_model_registry()
    overrides = model_overrides or {}
    classifier_client = registry.get_client("classifier", overrides)
    generator_client = registry.get_client("generator", overrides)
    doc_gen_client = registry.get_client("doc_generator", overrides)
    evaluator = get_evaluator(overrides.get("evaluator"))

    # ── Step 0: 分类（Classifier Agent, flash 模型）──
    _emit(progress_callback, {
        "type": "agent_status", "stage": "classifying",
        "content": "正在分析问题复杂度...",
    })
    state.classification = classify(user_query, classifier_client)
    state.max_rounds = state.classification.max_rounds
    state.timings["classify"] = (time.perf_counter() - t0) * 1000

    # 前端强制联网搜索：升权为 DEEP，确保 web_search 工具可用
    if force_web_search and state.classification.query_class != QueryClass.CHITCHAT:
        state.classification = ClassificationResult(
            query_class=QueryClass.DEEP,
            confidence=0.9,
            reason="force_web_search from frontend",
            max_rounds=3,
            max_tool_calls_per_round=2,
            allowed_tool_categories=["search", "analysis", "generation", "mcp"],
            degradation_max_level=2,
        )
        state.max_rounds = 3
        logger.info("Agent: force_web_search → classification overridden to DEEP")

    _emit(progress_callback, {
        "type": "agent_status", "stage": "classified",
        "content": f"问题分类: {state.classification.query_class.value}（{state.classification.reason}）",
        "class": state.classification.query_class.value,
        "max_rounds": state.max_rounds,
        "allowed_tools": state.classification.allowed_tool_categories,
    })

    # 闲聊直接返回（Generator Agent, pro 模型）
    if state.classification.query_class == QueryClass.CHITCHAT:
        _emit(progress_callback, {
            "type": "agent_status", "stage": "answering",
            "content": "闲聊模式，直接回答...",
        })
        state.final_answer = generator_client.chat([
            {"role": "system", "content": "你是 Mini Agent，一个企业项目知识助手。请友好、简洁地回答。"},
            {"role": "user", "content": user_query},
        ])
        state.timings["total"] = (time.perf_counter() - t0) * 1000
        logger.info("Agent: chitchat → direct answer (%.0fms)", state.timings["total"])
        return state

    tools = get_tool_specs_for_class(state.classification)
    if not state.allow_web_search:
        tools = [t for t in tools if t.get("function", {}).get("name") != "web_search"]
    logger.info(
        "Agent start | q=%.60s class=%s max_rounds=%d tools=%d",
        user_query, state.classification.query_class.value,
        state.max_rounds, len(tools),
    )

    # ── Round 1: 基础检索 ──
    state.round = 1
    _emit(progress_callback, {
        "type": "agent_status", "stage": "searching", "round": 1,
        "content": "Round 1: 正在检索项目文档...",
        "tool": "search_knowledge_base",
    })
    t_round = time.perf_counter()

    search_results, fusion_info = search_fn(user_query)
    state.fusion_info = fusion_info
    state.search_results = search_results
    state.all_results = list(search_results)
    state.seen_chunk_uids = {r.get("chunk_uid", "") for r in search_results}
    state.timings["round_1_search"] = (time.perf_counter() - t_round) * 1000

    # 记录工具调用（fusion_info 可能为 None，安全访问）
    fi = fusion_info or {}
    state.tool_call_history.append({
        "round": 1,
        "name": "search_knowledge_base",
        "arguments": json.dumps({"query": user_query}),
        "result_summary": f"命中 {len(search_results)} 条, "
                          f"faiss_ok={fi.get('faiss_ok')}, "
                          f"es_ok={fi.get('es_ok')}",
    })

    _emit(progress_callback, {
        "type": "agent_status", "stage": "searched", "round": 1,
        "content": f"Round 1 检索完成：命中 {len(search_results)} 条片段 "
                   f"(FAISS={'可用' if fi.get('faiss_ok') else '不可用'}, "
                   f"ES={'可用' if fi.get('es_ok') else '不可用'}, "
                   f"耗时 {state.timings.get('round_1_search', 0):.0f}ms)",
        "results_count": len(search_results),
        "fusion_info": {
            "faiss_ok": fi.get("faiss_ok"),
            "es_ok": fi.get("es_ok"),
            "concentration": fi.get("faiss_conc", fi.get("es_conc", 0)),
        },
    })

    # ── 强制文档生成（检测到意图时跳过 LLM 工具选择，直接执行）──
    # 必须在降级链检查之前执行，因为用户请求文档生成时即使检索无结果也应生成文档
    doc_gen_tool = _detect_doc_gen_intent(user_query, state.classification)
    if doc_gen_tool and not state.doc_gen_done:
        tool_name, tool_args = doc_gen_tool
        _emit(progress_callback, {
            "type": "agent_status", "stage": "tool_exec", "round": 1,
            "content": f"检测到文档生成意图 → 强制执行 {tool_name}...",
            "tool": tool_name,
        })
        logger.info("Agent: forced doc gen | tool=%s query=%.50s", tool_name, user_query)
        doc_results, doc_summary = _execute_tool(
            tool_name, tool_args, state, generator_client, search_fn, doc_gen_client,
        )
        state.tool_call_history.append({
            "round": 1,
            "name": tool_name,
            "arguments": json.dumps(tool_args, ensure_ascii=False),
            "result_summary": doc_summary,
        })
        if doc_results:
            for r in doc_results:
                cuid = r.get("chunk_uid", "")
                if cuid not in state.seen_chunk_uids:
                    state.seen_chunk_uids.add(cuid)
                    state.all_results.append(r)
            state.doc_gen_done = True  # 仅在成功生成时标记
        else:
            logger.warning("Agent: forced doc gen %s produced no results, not marking doc_gen_done", tool_name)
        _emit(progress_callback, {
            "type": "agent_status", "stage": "tool_done", "round": 1,
            "content": f"{tool_name} 完成 — {doc_summary}",
            "tool": tool_name,
            "result_summary": doc_summary,
        })

    # ── 本地检索质量不足时的处理（统一 QualitySignal） ──
    # empty:        零命中 → 先尝试跨语言改写
    # poor + DEEP:  有结果但集中度<0.3（分散在无关文档中）→ 给一次查询改写重试
    # poor + SHALLOW: 直接降级（SHALLOW 不进 Agent 循环，多轮检索无意义）
    quality_signal = QualitySignal.from_fusion(fusion_info, len(search_results))

    if quality_signal.should_degrade_by_concentration and not state.doc_gen_done:
        # Issue 3: empty 跳过改写重试和 Evaluator（索引无内容，改写无用），直接降级
        # poor 仅 DEEP 重试（DEEP 有后续多轮优化空间）
        needs_retry = (
            not quality_signal.is_empty
            and quality_signal.is_poor
            and state.classification.query_class == QueryClass.DEEP
        )
        if needs_retry:
            retry_reason = "empty results" if quality_signal.is_empty else f"poor quality (concentration={quality_signal.concentration:.2f})"
            rewritten_query = _try_cross_lingual_rewrite(generator_client, user_query)
            if rewritten_query and rewritten_query != user_query:
                logger.info("Agent: %s → trying rewrite: %.50s", retry_reason, rewritten_query)
                _emit(progress_callback, {
                    "type": "agent_status", "stage": "searching", "round": 1,
                    "content": f"检索质量不足（{quality_signal.retrieval_grade}），尝试查询改写: {rewritten_query[:80]}...",
                    "tool": "rewrite_query",
                })
                t_retry = time.perf_counter()
                retry_results, retry_fusion = search_fn(rewritten_query)
                retry_signal = QualitySignal.from_fusion(retry_fusion, len(retry_results))
                state.timings["round_1_rewrite_retry"] = (time.perf_counter() - t_retry) * 1000
                # 重试后质量达到 marginal 或 good → 拯救成功
                if not retry_signal.should_degrade_by_concentration:
                    logger.info("Agent: rewrite retry hit %d results (grade=%s) → continue", len(retry_results), retry_signal.retrieval_grade)
                    state.fusion_info = retry_fusion
                    state.search_results = retry_results
                    state.all_results = list(retry_results)
                    state.seen_chunk_uids = {r.get("chunk_uid", "") for r in retry_results}
                    state.tool_call_history.append({
                        "round": 1,
                        "name": "rewrite_query",
                        "arguments": json.dumps({"original_query": user_query, "rewritten": rewritten_query}),
                        "result_summary": f"查询改写命中 {len(retry_results)} 条 (grade={retry_signal.retrieval_grade})",
                    })
                    _emit(progress_callback, {
                        "type": "agent_status", "stage": "searched", "round": 1,
                        "content": f"查询改写后命中 {len(retry_results)} 条，继续分析...",
                        "results_count": len(retry_results),
                    })
                    quality_signal = retry_signal  # 更新信号，跳过后续降级判断

        # 重试后仍不足（或 SHALLOW+Poor 不重试直接到此）→ 给 Evaluator 二次判断机会
        if quality_signal.should_degrade_by_concentration:
            # Issue 6: 改写重试失败后，用 Evaluator 判断浓度低是否真的意味着内容不相关
            if not quality_signal.is_empty and len(search_results) > 0:
                completeness = evaluator.quick_completeness_check(user_query, search_results)
                logger.info("Agent: grade=%s → completeness check=%d before degrading", quality_signal.retrieval_grade, completeness)
                if completeness >= COMPLETENESS_MINIMUM:
                    # LLM 判断内容相关！concentration 低只因结果分散但语义相关 → 标记 marginal 继续
                    quality_signal.merge_llm_assessment(completeness=completeness, groundedness=0)
                    quality_signal.concentration = max(quality_signal.concentration, CONCENTRATION_MARGINAL + 0.01)
                    _emit(progress_callback, {
                        "type": "agent_status", "stage": "quality_salvaged",
                        "content": f"Evaluator 判断内容可用（完整度={completeness}%），浓度虽低但语义相关，继续 RAG 流程...",
                    })
                else:
                    # 真的不相关 → 降级
                    if state.force_web_search or (not state.scenario_matches and state.allow_web_search):
                        logger.info("Agent: grade=%s completeness=%d → auto web search", quality_signal.retrieval_grade, completeness)
                        _emit(progress_callback, {
                            "type": "agent_status", "stage": "web_searching",
                            "content": "本地文档未找到相关内容，自动联网搜索...",
                        })
                        state.final_answer = _generate_web_search_answer(
                            generator_client, user_query, state,
                        )
                        state.suppress_sources = True
                        state.timings["total"] = (time.perf_counter() - t0) * 1000
                        return state
                    else:
                        state.degradation_triggered = True
                        state.degradation_result = build_degradation_answer(
                            generator_client, user_query,
                            max_level=state.classification.degradation_max_level if state.allow_web_search else 1,
                        )
                        state.final_answer = format_degradation_response(state.degradation_result)
                        state.suppress_sources = True
                        state.timings["total"] = (time.perf_counter() - t0) * 1000
                        return state
            else:
                # 完全空结果 → 降级或自动联网搜索
                if state.force_web_search or (state.allow_web_search and state.classification.degradation_max_level >= 2):
                    logger.info("Agent: empty results → auto web search (allow_web_search=%s)", state.allow_web_search)
                    _emit(progress_callback, {
                        "type": "agent_status", "stage": "web_searching",
                        "content": "本地文档未找到相关内容，自动联网搜索...",
                    })
                    state.final_answer = _generate_web_search_answer(
                        generator_client, user_query, state,
                    )
                    state.suppress_sources = True
                    state.timings["total"] = (time.perf_counter() - t0) * 1000
                    return state

                logger.info("Agent: grade=%s → degradation chain (max_level=%d)", quality_signal.retrieval_grade, state.classification.degradation_max_level)
                _emit(progress_callback, {
                    "type": "agent_status", "stage": "degrading",
                    "content": "文档库未找到相关内容...",
                    "level": state.classification.degradation_max_level,
                })
                state.degradation_triggered = True
                state.degradation_result = build_degradation_answer(
                    generator_client, user_query,
                    max_level=state.classification.degradation_max_level if state.allow_web_search else 1,
                )
                state.final_answer = format_degradation_response(state.degradation_result)
                state.suppress_sources = True
                state.timings["total"] = (time.perf_counter() - t0) * 1000
                return state

    # 保存初始质量信号供后续使用
    state.quality_signal = quality_signal
    # ── end of quality gate ──

    # ── 文档生成后：Evaluator 完整性检查 → 生成回答（不再跳过评判）──
    if state.doc_gen_done:
        all_ctx = state.all_results if state.all_results else search_results
        completeness = evaluator.quick_completeness_check(user_query, all_ctx)

        if completeness < COMPLETENESS_MINIMUM:
            logger.info("Agent: doc_gen completeness=%d < %d → degradation", completeness, COMPLETENESS_MINIMUM)
            _emit(progress_callback, {
                "type": "agent_status", "stage": "degrading",
                "content": f"文档已生成但内容不足（完整度={completeness}%），尝试补充...",
                "level": state.classification.degradation_max_level,
            })
            state.degradation_triggered = True
            state.degradation_result = build_degradation_answer(
                generator_client, user_query,
                max_level=state.classification.degradation_max_level if state.allow_web_search else 1,
            )
            state.final_answer = format_degradation_response(state.degradation_result)
            state.suppress_sources = True
            state.timings["total"] = (time.perf_counter() - t0) * 1000
            return state

        _emit(progress_callback, {
            "type": "agent_status", "stage": "answering",
            "content": f"文档生成完成（完整度={completeness}%），正在整理回答...",
        })
        state.final_answer = _generate_final_answer(
            generator_client, state, user_query, all_ctx,
        )
        state.timings["total"] = (time.perf_counter() - t0) * 1000
        return state

    # ── Shallow 查询：跳过 Agent 循环，Evaluator 完整性检查后生成回答 ──
    if state.classification.query_class == QueryClass.SHALLOW:
        results_for_answer = state.all_results if state.all_results else search_results
        completeness = evaluator.quick_completeness_check(user_query, results_for_answer)

        if completeness < COMPLETENESS_MINIMUM:
            logger.info("Agent: shallow completeness=%d < %d → degradation", completeness, COMPLETENESS_MINIMUM)
            _emit(progress_callback, {
                "type": "agent_status", "stage": "degrading",
                "content": f"检索结果不足以回答（完整度={completeness}%），尝试其他方式...",
                "level": state.classification.degradation_max_level,
            })
            state.degradation_triggered = True
            state.degradation_result = build_degradation_answer(
                generator_client, user_query,
                max_level=state.classification.degradation_max_level if state.allow_web_search else 1,
            )
            state.final_answer = format_degradation_response(state.degradation_result)
            state.suppress_sources = True
            state.timings["total"] = (time.perf_counter() - t0) * 1000
            return state

        logger.info("Agent: shallow query → completeness=%d, generate answer directly", completeness)
        _emit(progress_callback, {
            "type": "agent_status", "stage": "answering",
            "content": f"检索完成 ({len(search_results)} 条结果，完整度={completeness}%)，正在生成回答...",
        })
        state.quality_signal = quality_signal.merge_llm_assessment(completeness=completeness, groundedness=0)
        state.final_answer = _generate_final_answer(
            generator_client, state, user_query, results_for_answer,
        )
        state.timings["total"] = (time.perf_counter() - t0) * 1000
        return state

    # ── Round 1 Critic（仅 deep 查询, Evaluator Agent, flash 模型）──
    # Issue 1 优化：concentration >= 0.5 且结果充足时，跳过 Evaluator LLM 调用
    if quality_signal.is_good and len(search_results) >= 10:
        logger.info("Agent: DEEP Round 1 quality already good (conc=%.2f results=%d), skip Evaluator",
                    quality_signal.concentration, len(search_results))
        quality_signal.merge_llm_assessment(
            completeness=75, groundedness=70,
            decision="STOP", decision_reason="signal: good concentration + sufficient results (fast path)",
        )
        _emit(progress_callback, {
            "type": "agent_status", "stage": "critic_done", "round": 1,
            "content": f"检索质量信号良好（concentration={quality_signal.concentration:.2f}，{len(search_results)} 条结果），跳过评估...",
            "retrieval_quality": "good", "completeness": 75, "decision": "STOP",
        })
    else:
        _emit(progress_callback, {
            "type": "agent_status", "stage": "critic", "round": 1,
            "content": "正在评估检索质量...",
        })
        # EvaluatorAgent 独立评判（flash 模型，非生成模型自评）
        quality_signal.info_gain = 1.0
        quality_signal = evaluator.assess(
            user_query, search_results, quality_signal,
        )
        _emit(progress_callback, {
            "type": "agent_status", "stage": "critic_done", "round": 1,
            "content": f"质量评估: {quality_signal.retrieval_grade}, "
                       f"完整度={quality_signal.completeness}%, 决策={quality_signal.decision}",
            "retrieval_quality": quality_signal.retrieval_grade,
            "completeness": quality_signal.completeness,
            "groundedness": quality_signal.groundedness,
            "decision": quality_signal.decision,
        })

    # 保存质量信号 + Critic 历史
    state.quality_signal = quality_signal
    state.critic_history.append(CriticAssessment(
        retrieval_quality=quality_signal.retrieval_grade,
        faiss_ok=quality_signal.faiss_ok,
        concentration=quality_signal.concentration,
        info_gain=quality_signal.info_gain,
        completeness=quality_signal.completeness,
        groundedness=quality_signal.groundedness,
        uncertainties=quality_signal.uncertainties,
        decision=quality_signal.decision,
        decision_reason=quality_signal.decision_reason,
        uncertainty_handling=quality_signal.uncertainty_handling,
    ))
    if quality_signal.uncertainties:
        existing = set(state.uncertainties)
        for u in quality_signal.uncertainties:
            if u not in existing:
                state.uncertainties.append(u)
                existing.add(u)

    _update_termination_counters_from_signal(state, quality_signal)
    state.consecutive_non_search_rounds = 0
    state.consecutive_low_gain_rounds = 0

    # Issue 1 优化: Round 1 质量已充分 → 直接生成回答（跳过 Agent 循环）
    if quality_signal.completeness >= 70 and quality_signal.is_good and len(search_results) >= 8:
        logger.info("Agent: sufficient quality after Round 1 (comp=%d conc=%.2f) → fast answer",
                    quality_signal.completeness, quality_signal.concentration)
        _emit(progress_callback, {
            "type": "agent_status", "stage": "answering",
            "content": f"检索质量充分（完整度={quality_signal.completeness}%），直接生成回答...",
        })
        state.final_answer = _generate_final_answer(
            generator_client, state, user_query, search_results,
        )
        state.timings["total"] = (time.perf_counter() - t0) * 1000
        return state

    should_stop, stop_reason = _check_termination(state)
    if should_stop:
        logger.info("Agent: termination after Round 1 → %s", stop_reason)
        _emit(progress_callback, {
            "type": "agent_status", "stage": "terminating",
            "content": f"终止条件触发: {stop_reason}，正在生成回答...",
            "reason": stop_reason,
        })
        _emit(progress_callback, {
            "type": "agent_status", "stage": "answering",
            "content": "正在基于检索结果生成回答...",
        })
        state.final_answer = _generate_final_answer(
            generator_client, state, user_query, search_results,
        )
        state.timings["total"] = (time.perf_counter() - t0) * 1000
        return state

    # ── Agent 循环（Rounds 2-N） ──
    while state.round < state.max_rounds:
        state.round += 1
        logger.info(
            "Agent Round %d/%d | grade=%s conc=%.2f gain=%.2f",
            state.round, state.max_rounds, quality_signal.retrieval_grade, quality_signal.concentration, state.last_info_gain,
        )

        _emit(progress_callback, {
            "type": "agent_status", "stage": "thinking", "round": state.round,
            "content": f"Round {state.round}/{state.max_rounds}: Agent 分析检索信号，选择下一步策略...",
            "prev_quality": quality_signal.retrieval_grade,
            "prev_concentration": quality_signal.concentration,
            "prev_gain": state.last_info_gain,
        })

        # Think: 让 LLM 选择工具（Generator Agent, pro 模型）
        agent_messages = _build_agent_messages(state, initial_search_done=True)

        tool_choice = None
        tool_result = _call_llm_for_tool(generator_client, agent_messages, tools, tool_choice)

        if tool_result is None:
            logger.warning("Agent Round %d: LLM returned no tool call, assuming answer phase", state.round)
            _emit(progress_callback, {
                "type": "agent_status", "stage": "no_tool", "round": state.round,
                "content": f"Round {state.round}: Agent 未选择工具，进入回答阶段",
            })
            # 统一通过 _update_termination_counters 更新所有计数器，避免 L3/L4 计数器残留
            state.last_info_gain = 0.0  # 无工具调用 = 无新信息
            _update_termination_counters(state, None)
            should_stop, stop_reason = _check_termination(state)
            if should_stop:
                break
            continue

        tool_name = tool_result.get("name", "")
        tool_args = tool_result.get("arguments", {})

        _emit(progress_callback, {
            "type": "agent_status", "stage": "tool_exec", "round": state.round,
            "content": f"Round {state.round}: 执行 {tool_name}...",
            "tool": tool_name,
        })

        # Act: 执行工具
        new_results, result_summary = _execute_tool(
            tool_name, tool_args, state, generator_client, search_fn, doc_gen_client,
        )

        _emit(progress_callback, {
            "type": "agent_status", "stage": "tool_done", "round": state.round,
            "content": f"Round {state.round}: {tool_name} 完成 — {result_summary}",
            "tool": tool_name,
            "result_summary": result_summary,
            "new_results": len(new_results),
        })

        state.tool_call_history.append({
            "round": state.round,
            "name": tool_name,
            "arguments": json.dumps(tool_args, ensure_ascii=False),
            "result_summary": result_summary,
        })

        # Observe: 更新状态（search_results 和 all_results 同步去重）
        if new_results:
            state.last_info_gain = compute_info_gain(new_results, state.seen_chunk_uids)
            for r in new_results:
                cuid = r.get("chunk_uid", "")
                if cuid not in state.seen_chunk_uids:
                    state.seen_chunk_uids.add(cuid)
                    state.search_results.append(r)
                    state.all_results.append(r)

            # 更新 fusion_info（如果是检索类工具）
            if get_tool_category(tool_name) == "search" and "fusion_info" in tool_result:
                state.fusion_info = tool_result.get("fusion_info", state.fusion_info)
        else:
            state.last_info_gain = 0.0

        # Critic
        _emit(progress_callback, {
            "type": "agent_status", "stage": "critic", "round": state.round,
            "content": f"Round {state.round}: 评估本轮检索效果...",
        })
        # In-loop Critic: EvaluatorAgent (flash 模型)
        quality_signal.info_gain = state.last_info_gain
        quality_signal.results_count = len(state.search_results)
        quality_signal = evaluator.assess(
            user_query, state.search_results[-10:], quality_signal,
            existing_uncertainties=state.uncertainties,
        )
        state.quality_signal = quality_signal
        state.critic_history.append(CriticAssessment(
            retrieval_quality=quality_signal.retrieval_grade,
            faiss_ok=quality_signal.faiss_ok,
            concentration=quality_signal.concentration,
            info_gain=quality_signal.info_gain,
            completeness=quality_signal.completeness,
            groundedness=quality_signal.groundedness,
            uncertainties=quality_signal.uncertainties,
            decision=quality_signal.decision,
            decision_reason=quality_signal.decision_reason,
            uncertainty_handling=quality_signal.uncertainty_handling,
        ))
        # 累积不确定项（去重），而非每轮覆盖
        if quality_signal.uncertainties:
            existing = set(state.uncertainties)
            for u in quality_signal.uncertainties:
                if u not in existing:
                    state.uncertainties.append(u)
                    existing.add(u)

        _emit(progress_callback, {
            "type": "agent_status", "stage": "critic_done", "round": state.round,
            "content": f"Round {state.round} 评估: 质量={quality_signal.retrieval_grade}, "
                       f"完整度={quality_signal.completeness}%, 增益={state.last_info_gain:.0%}, 决策={quality_signal.decision}",
            "retrieval_quality": quality_signal.retrieval_grade,
            "completeness": quality_signal.completeness,
            "groundedness": quality_signal.groundedness,
            "decision": quality_signal.decision,
            "info_gain": state.last_info_gain,
        })

        # 更新终止计数器
        _update_termination_counters_from_signal(state, quality_signal)
        should_stop, stop_reason = _check_termination(state)

        logger.info(
            "Agent Round %d done | tool=%s gain=%.2f evaluator=%s stop=%s",
            state.round, tool_name, state.last_info_gain, quality_signal.decision, should_stop,
        )

        if should_stop:
            logger.info("Agent: termination → %s", stop_reason)
            _emit(progress_callback, {
                "type": "agent_status", "stage": "terminating",
                "content": f"终止条件触发: {stop_reason}",
                "reason": stop_reason,
                "round": state.round,
            })
            break

    # ── 最终回答由调用方负责生成（非流式调用 _generate_final_answer，流式调用 _generate_final_answer_stream）──
    # 此处仅记录状态，避免流式路径重复生成两次
    _emit(progress_callback, {
        "type": "agent_status", "stage": "answering",
        "content": f"共 {state.round} 轮检索，累计 {len(state.all_results)} 条结果，正在生成最终回答...",
        "total_rounds": state.round,
        "total_results": len(state.all_results),
    })
    state.timings["total"] = (time.perf_counter() - t0) * 1000

    logger.info(
        "Agent done | rounds=%d class=%s time=%.0fms results=%d",
        state.round, state.classification.query_class.value,
        state.timings["total"], len(state.all_results),
    )

    return state


def _call_llm_for_tool(
    llm_client,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str | None = None,
) -> dict | None:
    """调用 LLM 获取工具选择。

    优先使用 function calling API，若 LLM 返回文本而非工具调用，
    降级到 JSON prompt 模式再试一次（确保工具调用不被跳过）。
    返回 {"name": str, "arguments": dict} 或 None。
    """
    # 尝试 function calling
    if hasattr(llm_client, 'chat_with_tools'):
        try:
            response = llm_client.chat_with_tools(
                messages, tools,
                tool_choice=tool_choice or "auto",
                temperature=0.3,
            )
            tool_calls = response.get("tool_calls")
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                return {"name": tc["function"]["name"], "arguments": args}

            # 没有工具调用 → LLM 可能忽略了工具，尝试 prompt 降级
            if response.get("content"):
                logger.info("LLM returned text instead of tool call: %.100s, trying prompt fallback", response["content"])

        except Exception:
            logger.exception("Function calling failed, falling back to prompt mode")

    # 降级：prompt 模式
    try:
        tool_list = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in tools
        )
        prompt = (
            f"{messages[-1]['content']}\n\n"
            f"可用工具：\n{tool_list}\n\n"
            f"请选择最合适的工具，输出 JSON："
            f'{{"tool": "工具名", "arguments": {{}}}}'
        )
        result = llm_client.chat_json([
            {"role": "system", "content": "你是工具选择器，必须选择并调用一个工具。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ], temperature=0.1)
        if result:
            return {
                "name": result.get("tool", ""),
                "arguments": result.get("arguments", {}),
            }
    except Exception:
        logger.exception("Prompt-based tool selection failed")

    return None


def _execute_tool(
    tool_name: str,
    tool_args: dict,
    state: AgentState,
    llm_client,
    search_fn,
    doc_gen_client=None,
) -> tuple[list[dict], str]:
    """执行工具调用。

    多Agent架构：
    - llm_client (pro): 内容整合、查询改写、标题生成
    - doc_gen_client (flash): 文档格式化生成（PPT/Word/PDF/Excel/图）

    返回 (new_results, result_summary)。
    """
    summary = ""
    results: list[dict] = []

    try:
        if tool_name == "search_knowledge_base":
            query = tool_args.get("query", state.user_query)
            results, fusion = search_fn(query)
            state.fusion_info = fusion
            summary = f"命中 {len(results)} 条"

        elif tool_name == "rewrite_query":
            original = tool_args.get("original_query", state.user_query)
            feedback = tool_args.get("retrieval_feedback", "")
            rewritten = _llm_rewrite_query(llm_client, original, feedback)
            results, fusion = search_fn(rewritten)
            state.fusion_info = fusion
            summary = f"改写为 '{rewritten[:50]}...' → 命中 {len(results)} 条"

        elif tool_name == "multi_query_search":
            query = tool_args.get("query", state.user_query)
            variants = _llm_multi_query(llm_client, query)
            all_results: dict[str, dict] = {}
            for v in variants[:3]:
                r, _ = search_fn(v)
                for item in r:
                    cuid = item.get("chunk_uid", "")
                    if cuid not in all_results:
                        all_results[cuid] = item
            results = list(all_results.values())
            summary = f"Multi-Query ({len(variants)}个变体) → 合并命中 {len(results)} 条"

        elif tool_name == "hyde_search":
            query = tool_args.get("query", state.user_query)
            hyde_text = _llm_hyde(llm_client, query)
            results, fusion = search_fn(hyde_text)
            state.fusion_info = fusion
            summary = f"HyDE 生成 {len(hyde_text)} 字假想文档 → 命中 {len(results)} 条"

        elif tool_name == "rerank_results":
            query = tool_args.get("query", state.user_query)
            count = tool_args.get("result_count", 15)
            results, summary = _rerank_results(llm_client, state.all_results, query, count)

        elif tool_name == "decompose_task":
            query = tool_args.get("query", state.user_query)
            sub_queries = _llm_decompose(llm_client, query)
            all_results = {}
            for sq in sub_queries[:5]:
                r, _ = search_fn(sq)
                for item in r:
                    cuid = item.get("chunk_uid", "")
                    if cuid not in all_results:
                        all_results[cuid] = item
            results = list(all_results.values())
            summary = f"拆解为 {len(sub_queries)} 个子问题 → 合并命中 {len(results)} 条"

        elif tool_name == "analyze_retrieval_signals":
            # 纯分析工具，不返回新结果
            cos = tool_args.get("faiss_cosine", 0)
            conc = tool_args.get("concentration", 0)
            quality = "good" if conc > 0.5 else ("marginal" if conc > 0.3 else "poor")
            summary = f"信号分析: cosine={cos:.2f} concentration={conc:.2f} → {quality}"

        elif tool_name == "generate_answer":
            # 标记为回答阶段，由外层生成
            summary = "进入回答阶段"

        elif tool_name == "search_scenario_kb":
            query = tool_args.get("query", state.user_query)
            try:
                from app.services.scenario_matcher import get_scenario_matcher
                from app.core.database import SessionLocal
                matcher = get_scenario_matcher()
                sc_db = SessionLocal()
                try:
                    sc_result = matcher.match(
                        db=sc_db,
                        query=query,
                        top_k=5,
                        threshold=0.45,  # Agent 主动检索时用更宽松的阈值
                    )
                    sc_entries = sc_result.get("entries", [])
                    if sc_entries:
                        # 将场景匹配结果转为类似 chunk 的格式，同时更新 state.scenario_matches
                        state.scenario_matches = sc_entries
                        results = [{
                            "chunk_uid": f"scenario_{e['entry_uid']}",
                            "document_uid": "scenario_kb",
                            "filename": f"[知识库] {e.get('template_name', '')} — {e['title']}",
                            "content": _format_scenario_matches([e]),
                            "section_title": e.get('template_name', ''),
                            "page_no": 0,
                            "rrf_score": e.get('similarity_score', 0.5),
                        } for e in sc_entries]
                        summary = f"场景知识库命中 {len(sc_entries)} 条排障卡片"
                    else:
                        summary = "场景知识库未找到匹配的排障知识"
                finally:
                    sc_db.close()
            except Exception:
                summary = "场景知识库检索失败"
                logger.exception("Scenario KB search tool failed")

        elif tool_name == "web_search":
            query = tool_args.get("query", state.user_query)
            if not state.allow_web_search:
                summary = "联网搜索未启用 / 需要登录"
                results = []
                return results, summary
            try:
                from app.services.web_search_service import search_web
                web_results = search_web(query)
                if web_results:
                    # 将网络搜索结果转换为类似 chunk 的格式
                    results = [{
                        "chunk_uid": f"web_{i}",
                        "document_uid": "web_search",
                        "filename": f"[网络搜索] {wr.get('title', '')}",
                        "content": wr.get("snippet", ""),
                        "section_title": wr.get("url", ""),
                        "page_no": 0,
                        "rrf_score": 0.5,
                    } for i, wr in enumerate(web_results)]
                    summary = f"联网搜索命中 {len(web_results)} 条"
                else:
                    summary = "联网搜索无结果或未启用"
            except Exception:
                summary = "联网搜索失败"
                logger.exception("Web search tool failed")

        elif tool_name in ("generate_ppt", "generate_word", "generate_pdf",
                           "generate_excel", "generate_diagram", "generate_markdown"):
            # 多Agent文档生成：标题整合用 pro，格式输出用 flash
            title = tool_args.get("title", "")
            context = _format_search_context_static(state.all_results) if state.all_results else ""
            fmt_client = doc_gen_client or llm_client  # flash 优先，回退 pro
            try:
                from app.services.document_generation_service import (
                    generate_ppt, generate_word, generate_pdf,
                    generate_excel, generate_diagram, generate_markdown,
                    DocResult, _generate_title,
                )
                # 生成概括性标题（Generator Agent, pro 模型做内容整合）
                doc_type_label = {
                    "generate_ppt": "PPT演示文稿", "generate_word": "Word文档",
                    "generate_pdf": "PDF文档", "generate_excel": "Excel表格",
                    "generate_diagram": "图表", "generate_markdown": "Markdown文档",
                }
                if not title:
                    title = _generate_title(llm_client, state.user_query,
                                           doc_type_label.get(tool_name, "文档"), context)

                # 文档格式化生成（Doc Generator Agent, flash 模型）
                if tool_name == "generate_ppt":
                    outline = tool_args.get("outline", "")
                    doc_result = generate_ppt(
                        fmt_client, state.user_query, title, outline, context)
                elif tool_name == "generate_word":
                    ct = tool_args.get("content_type", "技术方案")
                    doc_result = generate_word(
                        fmt_client, state.user_query, title, ct, context)
                elif tool_name == "generate_pdf":
                    doc_result = generate_pdf(
                        fmt_client, state.user_query, title, context)
                elif tool_name == "generate_excel":
                    st = tool_args.get("sheet_type", "数据汇总")
                    doc_result = generate_excel(
                        fmt_client, state.user_query, title, st, context)
                elif tool_name == "generate_markdown":
                    doc_result = generate_markdown(
                        fmt_client, state.user_query, title, context)
                else:  # generate_diagram
                    dt = tool_args.get("diagram_type", "flowchart")
                    desc = tool_args.get("description", state.user_query)
                    doc_result = generate_diagram(
                        fmt_client, state.user_query, dt, desc, context)

                if doc_result.error:
                    summary = f"{tool_name} 失败: {doc_result.error}"
                elif tool_name == "generate_diagram":
                    results = [{
                        "chunk_uid": f"diagram_{tool_name}",
                        "document_uid": "diagram_gen",
                        "filename": f"[图表] {title}",
                        "content": doc_result.content,
                        "section_title": title,
                        "page_no": 0,
                        "rrf_score": 1.0,
                    }]
                    summary = f"图表生成完成 ({tool_args.get('diagram_type', 'flowchart')})"
                else:
                    # PPT/Word/PDF/Excel/Markdown: 将下载链接加入结果
                    download_url = f"/api/documents/download/{doc_result.filename}"
                    fmts = {
                        "generate_ppt": "PPT", "generate_word": "Word",
                        "generate_pdf": "PDF", "generate_excel": "Excel",
                        "generate_markdown": "Markdown",
                    }
                    fmt_label = fmts.get(tool_name, "文档")
                    results = [{
                        "chunk_uid": f"doc_{tool_name}",
                        "document_uid": "doc_gen",
                        "filename": f"[{fmt_label}] {doc_result.filename}",
                        "content": (
                            f"已生成 {fmt_label} 文档: {doc_result.filename}\n"
                            f"下载链接: {download_url}"
                        ),
                        "section_title": title,
                        "page_no": 0,
                        "rrf_score": 1.0,
                    }]
                    summary = f"{tool_name} 完成 → {doc_result.filename}"
            except Exception as e:
                summary = f"{tool_name} 执行失败: {e}"
                logger.exception("%s failed", tool_name)

        else:
            summary = f"未知工具: {tool_name}"

    except Exception as e:
        logger.exception("Tool execution failed: %s", tool_name)
        summary = f"工具执行失败: {e}"

    return results, summary


# ── LLM 辅助函数 ──

def _try_cross_lingual_rewrite(llm_client, query: str) -> str:
    """检测中文查询中的实体名称，生成英文/拼音变体用于跨语言检索。

    例如 "安克创新" → "Anker Innovation"，"中控技术" → "Zhongkong Technology SUPCON"。
    仅当查询包含中文时才触发，返回改写后的查询或空字符串。
    """
    import re
    has_chinese = bool(re.search(r'[一-鿿]', query))
    if not has_chinese:
        return ""

    prompt = (
        "你是一个跨语言检索助手。用户用中文搜索，但文档库中的文档可能使用英文名称。\n"
        "请将用户查询中的中文实体名称（公司名、产品名、技术术语）翻译为英文或拼音，"
        "生成一个适合英文文档检索的查询字符串。\n"
        "只输出改写后的查询，不要解释。如果查询本身就是合适的检索词，原样返回。\n\n"
        f"用户查询: {query}\n"
        "改写查询:"
    )
    try:
        rewritten = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.2).strip()
        if rewritten and rewritten != query:
            logger.info("Cross-lingual rewrite: %.60s → %.60s", query, rewritten)
            return rewritten
    except Exception:
        logger.debug("Cross-lingual rewrite LLM call failed")
    return ""


def _llm_rewrite_query(llm_client, original: str, feedback: str = "") -> str:
    prompt = (
        f"改写以下检索查询以提升召回率。注意：如果原始查询包含中文实体名称（公司名/产品名/人名），"
        f"请同时尝试对应的英文/拼音变体，用空格分隔中英文关键词。\n"
        f"原始查询：{original}\n"
    )
    if feedback:
        prompt += f"检索反馈：{feedback}\n"
    prompt += "改写后查询："
    try:
        return llm_client.chat([{"role": "user", "content": prompt}], temperature=0.3).strip()
    except Exception:
        return original


def _llm_multi_query(llm_client, query: str) -> list[str]:
    prompt = (
        f"为以下问题生成 2-3 个不同视角的检索查询变体，每行一个，不要编号：\n"
        f"问题：{query}\n"
        f"变体查询："
    )
    try:
        resp = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.5).strip()
        variants = [q.strip() for q in resp.split("\n") if q.strip() and q.strip() != query]
        return list(dict.fromkeys(variants))[:3]
    except Exception:
        return [query]


def _llm_hyde(llm_client, query: str) -> str:
    prompt = (
        "你是一个技术文档助手。请用一段 80-150 字的技术文档片段风格来回答以下问题。"
        "以文档口吻写出，包含关键术语和概念，不要用对话语气。直接输出内容，不要铺垫。\n"
        f"问题: {query}\n"
        "文档片段:"
    )
    try:
        return llm_client.chat([{"role": "user", "content": prompt}], temperature=0.3).strip()
    except Exception:
        return query


def _llm_decompose(llm_client, query: str) -> list[str]:
    prompt = (
        "你是一个任务拆解助手。将以下复杂问题拆解为 2-5 个独立的子问题，"
        "每个子问题可以被独立检索和回答。每行一个子问题，不要编号。\n"
        f"复杂问题: {query}\n"
        "子问题:"
    )
    try:
        resp = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.3).strip()
        return [q.strip() for q in resp.split("\n") if q.strip()][:5]
    except Exception:
        return [query]


def _rerank_results(llm_client, results: list[dict], query: str, count: int) -> tuple[list[dict], str]:
    """Reranker 精排。尝试使用 RerankerService，失败则保持原序。"""
    try:
        from app.services.reranker_service import RerankerService
        reranker = RerankerService()
        if reranker.available:
            ranked = reranker.rerank(query, results[:30], top_k=count)
            return ranked, f"Reranker 精排 {len(results[:30])} → {len(ranked)} 条"
    except Exception:
        logger.exception("Reranker failed")
    return results[:count], f"Reranker 不可用，保持原序 {min(len(results), count)} 条"


def _format_scenario_matches(matches: list[dict]) -> str:
    """将场景知识库匹配结果格式化为 LLM 可读的结构化上下文。"""
    if not matches:
        return ""
    fragments = []
    for i, m in enumerate(matches, 1):
        frag = (
            f"### 排障场景 {i}: {m.get('template_name', '')} — {m.get('title', '')}\n"
            f"场景分类: {m.get('template_name', '未知')}\n"
            f"标签: {m.get('tags', '')}\n"
            f"匹配度: {m.get('similarity_score', 0):.0%}\n\n"
            f"知识内容:\n"
        )
        try:
            import json
            content = json.loads(m.get("content_json", "{}"))
            for key, value in content.items():
                if isinstance(value, list):
                    frag += f"**{key}**:\n"
                    for v in value:
                        frag += f"  - {v}\n"
                elif isinstance(value, str):
                    frag += f"**{key}**: {value}\n"
                elif isinstance(value, dict):
                    frag += f"**{key}**: {'; '.join(f'{k}={v}' for k, v in value.items())}\n"
            frag += "\n---\n"
        except Exception:
            frag += m.get("content_json", "") + "\n\n---\n"
        fragments.append(frag)
    return "\n".join(fragments)


def _format_search_context_static(results: list[dict]) -> str:
    """将搜索结果格式化为 LLM 可读的上下文字符串（静态版本，避免循环依赖）。"""
    if not results:
        return ""
    fragments = []
    appendix = []
    for i, r in enumerate(results, 1):
        fragments.append(f"[片段 {i}]\n{r.get('content', '')}")
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


def _filter_results_by_quality(
    results: list[dict], state: AgentState,
) -> list[dict]:
    """当场景知识库无匹配时，基于检索信号质量过滤低相关度结果。

    策略：
    - 文档集中度 >= 0.5：保留全部结果（检索质量良好）
    - 文档集中度 >= 0.3：过滤 rrf_score < 0.15 的尾部噪声
    - 文档集中度 < 0.3：仅保留 rrf_score >= 0.2 的结果，至少保留 top-5
    """
    import math
    if not results:
        return results

    conc = 0.0
    if state.fusion_info:
        conc = state.fusion_info.get("faiss_conc", state.fusion_info.get("es_conc", 0.0))
    conc = float(conc) if conc else 0.0

    if math.isnan(conc) or conc < 0:
        conc = 0.0

    if conc >= 0.5:
        return results

    if conc >= 0.3:
        score_threshold = 0.15
    else:
        score_threshold = 0.20

    filtered = []
    for r in results:
        score = r.get("rrf_score", r.get("faiss_score", r.get("es_score", 0)))
        if score >= score_threshold:
            filtered.append(r)

    if len(filtered) < 5 and results:
        results_sorted = sorted(
            results,
            key=lambda x: x.get("rrf_score", x.get("faiss_score", x.get("es_score", 0))),
            reverse=True,
        )
        filtered = results_sorted[:5]

    if len(filtered) < len(results):
        logger.info(
            "Quality filter: kept %d/%d results (concentration=%.2f, threshold=%.2f)",
            len(filtered), len(results), conc, score_threshold,
        )

    return filtered


def _filter_results_by_scenario_context(
    results: list[dict], state: AgentState,
) -> list[dict]:
    """当场景知识库高置信度匹配时，过滤掉与场景主题无关的文档结果。

    策略：提取场景匹配中的关键技术词，只保留文档结果中与这些词有交集的。
    若场景 KB 匹配度 >= 0.75，只保留 rrf_score >= 0.3 的文档结果。
    """
    if not results:
        return results

    # 从场景匹配中提取关键词
    scenario_keywords: set[str] = set()
    for m in state.scenario_matches:
        tags = m.get("tags", "")
        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                scenario_keywords.add(tag.lower())
        title = m.get("title", "")
        for word in title.replace("故障排查", "").split():
            if len(word) >= 2:
                scenario_keywords.add(word.lower())

    if not scenario_keywords:
        return results

    score_threshold = 0.2 if state.scenario_match_score >= 0.75 else 0.0

    filtered = []
    for r in results:
        content = (r.get("content", "") or "").lower()
        filename = (r.get("filename", "") or "").lower()
        section = (r.get("section_title", "") or "").lower()
        combined = f"{filename} {section} {content[:300]}"

        # 检查是否与场景关键词有交集
        keyword_match = any(kw in combined for kw in scenario_keywords)
        if not keyword_match:
            continue

        # 高置信度场景额外要求最低分数
        if score_threshold > 0:
            score = r.get("rrf_score", r.get("faiss_score", 0))
            if score < score_threshold:
                continue

        filtered.append(r)

    # 如果全部被过滤且场景KB有高置信度匹配，说明本地文档库确实不覆盖此话题。
    # 不保留兜底结果——保留不相关文档比空结果更糟糕（会让LLM引用错误来源）。
    if not filtered:
        logger.info(
            "Scenario filter removed all %d results (scenario=%.40s, keywords=%s), "
            "no local docs relevant to this question",
            len(results),
            state.scenario_matches[0].get("title", "") if state.scenario_matches else "",
            list(scenario_keywords)[:8],
        )
        # 高置信度场景：不需要无关文档污染回答
        if score_threshold > 0 or state.scenario_match_score >= 0.6:
            return []
        # 低置信度：保留少量结果避免完全空回答
        results_sorted = sorted(
            results,
            key=lambda x: x.get("rrf_score", x.get("faiss_score", 0)),
            reverse=True,
        )
        return results_sorted[:2]

    return filtered


def _format_web_search_results(web_results: list[dict]) -> str:
    """将网络搜索结果格式化为 LLM 可读的上下文。"""
    parts = []
    for i, wr in enumerate(web_results, 1):
        title = wr.get("title", "")
        snippet = wr.get("snippet", "")
        url = wr.get("url", "")
        parts.append(f"[结果 {i}] {title}\n{snippet}\n来源: {url}")
    return "\n\n".join(parts)


def _generate_web_search_answer(
    llm_client,
    user_query: str,
    state: AgentState,
) -> str:
    """联网搜索 + LLM 整合回答。

    直接执行联网搜索，将结果交给 LLM 生成带 [网络搜索] 标注的整合回答。
    如果联网搜索也无结果，回退到降级链（通用知识）。
    """
    if not state.allow_web_search:
        result = build_degradation_answer(llm_client, user_query, max_level=1)
        return format_degradation_response(result)

    from app.services.web_search_service import search_web

    logger.info("Web search answer: q=%.50s", user_query)
    web_results = search_web(user_query)

    if not web_results:
        logger.info("Web search returned no results → fallback to degradation Level 1")
        result = build_degradation_answer(llm_client, user_query, max_level=1)
        return format_degradation_response(result)

    web_source_results: list[dict] = []
    for idx, wr in enumerate(web_results, 1):
        title = (wr.get("title") or "").strip() or f"联网搜索结果 {idx}"
        url = (wr.get("url") or "").strip()
        snippet = (wr.get("snippet") or wr.get("summary") or wr.get("content") or "").strip()
        web_source_results.append({
            "chunk_uid": f"web_search_{idx}",
            "document_uid": "web_search",
            "filename": title,
            "section_title": url,
            "content": snippet,
            "page_no": 0,
            "rrf_score": max(0.0, 1.0 - idx * 0.01),
            "source_type": "web_search",
            "url": url,
            "domain": wr.get("domain") or "",
        })
    existing_urls = {r.get("section_title") or r.get("url") for r in state.all_results}
    for item in web_source_results:
        if (item.get("section_title") or item.get("url")) not in existing_urls:
            state.all_results.append(item)
            state.search_results.append(item)
    state.tool_call_history.append({
        "round": state.round,
        "name": "web_search",
        "query": user_query,
        "result_count": len(web_source_results),
    })

    web_context = _format_web_search_results(web_results)

    system_prompt = (
        "你是 Mini Agent，一个IT运维知识助手。\n\n"
        "项目文档库和场景知识库中未找到与用户问题直接相关的内容，"
        "以下信息来自**联网搜索**。\n\n"
        "规则：\n"
        "1. 基于联网搜索结果回答用户问题，优先使用搜索结果中的信息\n"
        "2. 所有来自网络搜索的内容标注 [网络搜索]\n"
        "3. 基于自身知识补充或推断的内容标注 [通用知识]\n"
        "4. 如果网络搜索结果也不足以全面回答，诚实说明不足之处\n"
        "5. 在回答开头用一句话说明信息来源（如：以下回答基于联网搜索结果）\n"
        "6. 不要在正文末尾列出参考来源、URL 清单或来源编号，系统会在来源追踪区展示网页来源"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"联网搜索结果：\n\n{web_context}"},
        {"role": "user", "content": user_query},
    ]

    try:
        answer = llm_client.chat(messages, temperature=0.5)
    except Exception:
        logger.exception("Web search answer generation failed")
        result = build_degradation_answer(llm_client, user_query, max_level=1)
        return format_degradation_response(result)

    return answer


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算（与 chunk_service._rough_token_count 逻辑一致）。"""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    english_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    numbers = len(re.findall(r"\d+(?:\.\d+)?", text))
    other_chars = len(re.findall(r"[^\s一-鿿A-Za-z0-9]", text))
    return chinese_chars + english_words + numbers + max(1, other_chars // 4)


def _generate_final_answer(
    llm_client,
    state: AgentState,
    user_query: str,
    results: list[dict],
) -> str:
    """生成最终回答。集成降级链逻辑 + 场景知识库。"""
    # 如果触发了降级，但有场景知识库匹配结果 → 跳过降级，用场景知识库回答
    if state.degradation_triggered and state.degradation_result:
        if not state.scenario_matches:
            return format_degradation_response(state.degradation_result)
        logger.info(
            "Suppressing degradation: scenario KB has %d matches, using those instead",
            len(state.scenario_matches),
        )

    # 构建检索上下文（场景知识库匹配成功时，过滤低相关度文档结果）
    if state.scenario_matches and state.scenario_match_score >= 0.6:
        original_count = len(results)
        filtered_results = _filter_results_by_scenario_context(results, state)
        if len(filtered_results) < original_count:
            logger.info(
                "Filtered out %d low-relevance document results (scenario KB confidence=%.2f)",
                original_count - len(filtered_results), state.scenario_match_score,
            )
            results = filtered_results
            # 场景过滤移除全部本地文档 → 本地文档与此问题无关
            if not filtered_results:
                state.suppress_sources = True
                logger.info(
                    "All %d local docs filtered by scenario context (scenario=%.40s), suppressing sources",
                    original_count,
                    state.scenario_matches[0].get("title", "") if state.scenario_matches else "",
                )
    else:
        # 场景知识库无匹配时，基于检索信号质量过滤
        results = _filter_results_by_quality(results, state)

    context = _format_search_context_static(results)

    # 不确定项
    uncertainty_note = ""
    if state.uncertainties:
        uncertainty_note = (
            "\n\n## 不确定项\n以下内容当前文档库未覆盖，")
        uncertainty_note += "基于通用知识补充，已在正文标注 [通用知识]：\n"
        uncertainty_note += "\n".join(f"- {u}" for u in state.uncertainties)

    brief = _has_doc_gen_results(results)
    system_prompt = _get_answer_system_prompt(brief_mode=brief)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # ── 场景知识库匹配结果（结构化知识，优先级最高） ──
    if state.scenario_matches:
        scenario_context = _format_scenario_matches(state.scenario_matches)
        messages.append({
            "role": "system",
            "content": f"以下是从运维场景知识库中匹配到的结构化排障知识（高精度，优先参考）：\n\n{scenario_context}",
        })

    if context:
        messages.append({
            "role": "system",
            "content": f"以下是从项目文档中检索到的相关片段：\n\n{context}",
        })

    agent_history = (
        f"用户问题：{user_query}\n"
        f"检索轮次：{state.round} 轮\n"
        f"累计检索结果：{len(results)} 条\n"
    )
    if state.uncertainties:
        agent_history += f"当前不确定项：{', '.join(state.uncertainties)}\n"
    if uncertainty_note:
        agent_history += uncertainty_note
    # ── 多源信息标注规则 ──
    has_scenario = bool(state.scenario_matches)
    has_docs = bool(results)
    has_web = state.force_web_search or any(
        r.get("document_uid") == "web_search" for r in results
    )

    source_rules = "\n\n来源标注规则（重要）：\n"
    if has_scenario:
        source_rules += "- 使用了「场景知识库」排障卡片的内容 → 标注 [知识库]\n"
    if has_docs:
        source_rules += "- 使用了「项目文档」检索片段的内容 → 标注 [文档]\n"
    source_rules += "- 基于通用知识推断的内容 → 标注 [通用知识]\n"
    if has_web:
        source_rules += "- 联网搜索结果 → 标注 [网络搜索]\n"
    if has_scenario and has_web:
        source_rules += (
            "\n**重要**：当前回答融合了多个信息来源（场景知识库 + 联网搜索"
        )
        if has_docs:
            source_rules += " + 项目文档"
        source_rules += (
            "）。请在回答开头用一句话说明信息来源，"
            "例如：'本回答综合了运维场景知识库、联网搜索和项目文档的信息。'\n"
        )
    elif has_scenario and has_docs:
        source_rules += "\n优先使用场景知识库的结构化排障内容，项目文档作为补充参考。\n"
    elif has_scenario:
        source_rules += "\n优先使用场景知识库的结构化排障内容。\n"

    agent_history += source_rules

    messages.append({"role": "user", "content": agent_history})

    # ── Token 用量统计（场景知识库 vs RAG vs System）──
    scenario_tokens = _estimate_tokens(_format_scenario_matches(state.scenario_matches)) if state.scenario_matches else 0
    doc_tokens = _estimate_tokens(context) if context else 0
    system_tokens = _estimate_tokens(messages[0].get("content", ""))
    total = sum(len(m.get("content", "") or "") for m in messages)
    logger.info(
        "Token estimate | system ~%d | scenario_kb ~%d (%d matches) | rag_docs ~%d (%d results) | total ~%d chars",
        system_tokens, scenario_tokens, len(state.scenario_matches), doc_tokens, len(results), total,
    )

    try:
        answer = llm_client.chat(messages, temperature=0.5)
    except Exception:
        logger.exception("Final answer generation failed")
        return "很抱歉，生成回答时出现错误。请重试。"

    # 检测 LLM 是否认为检索结果与问题无关 → 标记抑制来源展示
    if _answer_indicates_no_relevant_results(answer):
        state.suppress_sources = True
        logger.info("Final answer indicates no relevant results, suppressing sources")

    # 在回答末尾附加所有下载链接和 Mermaid 图表
    download_links = []
    diagram_blocks = []
    for r in results:
        content = r.get("content", "")
        if "/api/documents/download/" in content:
            url_match = content.split("下载链接: ", 1)
            if len(url_match) > 1:
                download_links.append(url_match[1].strip())
        elif r.get("chunk_uid", "").startswith("diagram_") and content:
            diagram_blocks.append(content)

    if download_links:
        answer += "\n\n" + "\n".join(f"下载链接: {link}" for link in download_links)
    for diag in diagram_blocks:
        answer += "\n\n```mermaid\n" + diag + "\n```\n"

    return answer


def _generate_final_answer_stream(
    llm_client,
    state: AgentState,
    user_query: str,
    results: list[dict],
):
    """流式生成最终回答。yield token 字符串。"""
    # 如果触发了降级，但有场景知识库匹配结果 → 跳过降级
    if state.degradation_triggered and state.degradation_result:
        if not state.scenario_matches:
            answer = format_degradation_response(state.degradation_result)
            chunk_size = 8
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i + chunk_size]
            return
        logger.info(
            "Suppressing degradation (stream): scenario KB has %d matches",
            len(state.scenario_matches),
        )

    context = _format_search_context_static(results)

    uncertainty_note = ""
    if state.uncertainties:
        uncertainty_note = (
            "\n\n## 不确定项\n以下内容当前文档库未覆盖，")
        uncertainty_note += "基于通用知识补充，已在正文标注 [通用知识]：\n"
        uncertainty_note += "\n".join(f"- {u}" for u in state.uncertainties)

    brief = _has_doc_gen_results(results)
    system_prompt = _get_answer_system_prompt(brief_mode=brief)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # ── 场景知识库匹配结果（结构化知识，优先级最高） ──
    if state.scenario_matches:
        scenario_context = _format_scenario_matches(state.scenario_matches)
        messages.append({
            "role": "system",
            "content": f"以下是从运维场景知识库中匹配到的结构化排障知识（高精度，优先参考）：\n\n{scenario_context}",
        })

    if context:
        messages.append({
            "role": "system",
            "content": f"以下是从项目文档中检索到的相关片段：\n\n{context}",
        })

    agent_history = (
        f"用户问题：{user_query}\n"
        f"检索轮次：{state.round} 轮\n"
        f"累计检索结果：{len(results)} 条\n"
    )
    if state.uncertainties:
        agent_history += f"当前不确定项：{', '.join(state.uncertainties)}\n"
    if uncertainty_note:
        agent_history += uncertainty_note
    # ── 多源信息标注规则 ──
    has_scenario = bool(state.scenario_matches)
    has_docs = bool(results)
    has_web = state.force_web_search or any(
        r.get("document_uid") == "web_search" for r in results
    )

    source_rules = "\n\n来源标注规则（重要）：\n"
    if has_scenario:
        source_rules += "- 使用了「场景知识库」排障卡片的内容 → 标注 [知识库]\n"
    if has_docs:
        source_rules += "- 使用了「项目文档」检索片段的内容 → 标注 [文档]\n"
    source_rules += "- 基于通用知识推断的内容 → 标注 [通用知识]\n"
    if has_web:
        source_rules += "- 联网搜索结果 → 标注 [网络搜索]\n"
    if has_scenario and has_web:
        source_rules += (
            "\n**重要**：当前回答融合了多个信息来源（场景知识库 + 联网搜索"
        )
        if has_docs:
            source_rules += " + 项目文档"
        source_rules += (
            "）。请在回答开头用一句话说明信息来源，"
            "例如：'本回答综合了运维场景知识库、联网搜索和项目文档的信息。'\n"
        )
    elif has_scenario and has_docs:
        source_rules += "\n优先使用场景知识库的结构化排障内容，项目文档作为补充参考。\n"
    elif has_scenario:
        source_rules += "\n优先使用场景知识库的结构化排障内容。\n"

    agent_history += source_rules

    messages.append({"role": "user", "content": agent_history})

    # ── Token 用量统计（流式路径）──
    total_chars_stream = sum(len(m.get("content", "") or "") for m in messages)
    logger.info(
        "Token estimate (stream) | system_prompt ~%d | scenario_kb ~%d (%d matches) | rag_docs ~%d (%d results) | total ~%d chars",
        _estimate_tokens(messages[0].get("content", "")),
        _estimate_tokens(_format_scenario_matches(state.scenario_matches)) if state.scenario_matches else 0,
        len(state.scenario_matches),
        _estimate_tokens(context) if context else 0,
        len(results),
        total_chars_stream,
    )

    full_answer = ""
    try:
        for token in llm_client.stream_chat(messages, temperature=0.5):
            full_answer += token
            yield token
    except Exception:
        logger.exception("Final answer stream failed")
        yield "很抱歉，生成回答时出现错误。请重试。"

    # 检测 LLM 是否认为检索结果与问题无关 → 标记抑制来源展示
    if _answer_indicates_no_relevant_results(full_answer):
        state.suppress_sources = True
        logger.info("Final answer stream indicates no relevant results, suppressing sources")

    # 在回答末尾附加所有下载链接和 Mermaid 图表
    download_links = []
    diagram_blocks = []
    for r in results:
        content = r.get("content", "")
        if "/api/documents/download/" in content:
            url_match = content.split("下载链接: ", 1)
            if len(url_match) > 1:
                download_links.append(url_match[1].strip())
        elif r.get("chunk_uid", "").startswith("diagram_") and content:
            diagram_blocks.append(content)

    if download_links:
        yield "\n\n" + "\n".join(f"下载链接: {link}" for link in download_links)
    for diag in diagram_blocks:
        yield "\n\n```mermaid\n" + diag + "\n```\n"


def _get_answer_system_prompt(brief_mode: bool = False) -> str:
    """获取回答阶段的 system prompt。brief_mode 用于文档生成后的精简回答。"""
    from pathlib import Path
    import json
    from app.core.config import settings
    from app.prompts.system_prompts import DEFAULT_SYSTEM_PROMPT

    # 加载 system_prompt：环境变量 → JSON 文件 → 代码默认值
    prompt = settings.system_prompt
    if prompt:
        base = prompt
    else:
        prompt_file = Path(settings.faiss_index_path).parent / "system_prompt.json"
        if prompt_file.exists():
            try:
                data = json.loads(prompt_file.read_text(encoding="utf-8"))
                base = data.get("prompt", DEFAULT_SYSTEM_PROMPT)
            except Exception:
                base = DEFAULT_SYSTEM_PROMPT
        else:
            base = DEFAULT_SYSTEM_PROMPT

    if brief_mode:
        base += (
            "\n\n## 当前模式：文档已生成\n"
            "用户要求的文档已经生成完毕。你的回答应该简短精炼：\n"
            "1. 用 2-3 句话总结文档的核心内容\n"
            "2. 提示用户点击下载链接获取完整文档\n"
            "3. 不要重复文档中的详细内容\n"
        )
    return base


def _has_doc_gen_results(results: list[dict]) -> bool:
    """检查结果中是否包含文档生成产物。"""
    for r in results:
        if r.get("chunk_uid", "").startswith("doc_") or r.get("chunk_uid", "").startswith("diagram_"):
            return True
        if "/api/documents/download/" in r.get("content", ""):
            return True
    return False


# LLM 回答中表示"检索结果与问题无关"的关键短语
_NO_RELEVANT_RESULTS_PATTERNS = [
    "未找到相关内容",
    "未包含与",
    "文档库中未",
    "文档库中没有",
    "文档中没有相关",
    "未找到与",
    "没有找到相关",
    "不包含相关",
    "无法从文档中找到",
    "没有相关的文档",
    "未检索到相关",
]

def _answer_indicates_no_relevant_results(answer: str) -> bool:
    """检测 LLM 回答是否明确表示检索结果与问题无关。"""
    if not answer:
        return False
    # 只在前 500 字符中检查（开头部分）避免误判
    head = answer[:500]
    return any(p in head for p in _NO_RELEVANT_RESULTS_PATTERNS)
