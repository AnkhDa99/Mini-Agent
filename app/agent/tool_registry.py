"""
工具注册表 + 本地工具实现。

工具按类别组织：
- search:    基础检索、Query Rewrite、Multi-Query、HyDE
- analysis:  Reranker精排、任务拆解、检索信号分析（仅 deep）
- generation: 生成最终回答
- mcp:       联网搜索、文档生成、图表（仅 deep，按需加载）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """OpenAI function calling 格式的工具定义。"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    category: str  # search | analysis | generation | mcp
    requires_deep: bool = False
    executor: Callable | None = None  # 实际执行函数


# ── 工具定义（OpenAI function calling JSON Schema） ──

TOOL_SEARCH_KNOWLEDGE_BASE = {
    "name": "search_knowledge_base",
    "description": "在项目文档库中检索相关内容。使用 Concentration-RRF 融合 FAISS 向量搜索和 ES 关键词搜索的结果。返回最相关的文档片段及其分数。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询文本。应该是对用户问题的精确表述。",
            },
        },
        "required": ["query"],
    },
}

TOOL_REWRITE_QUERY = {
    "name": "rewrite_query",
    "description": "用 LLM 改写检索查询，优化措辞以提升检索召回率。适用于初次检索结果质量不佳时（cosine < 0.7 或 concentration < 0.5）。",
    "parameters": {
        "type": "object",
        "properties": {
            "original_query": {
                "type": "string",
                "description": "原始查询文本。",
            },
            "retrieval_feedback": {
                "type": "string",
                "description": "上一轮检索的反馈信息：cosine分数、concentration值、覆盖的文档列表。",
            },
        },
        "required": ["original_query"],
    },
}

TOOL_MULTI_QUERY = {
    "name": "multi_query_search",
    "description": "生成 2-3 个不同视角的检索查询变体，多路检索后合并结果。适用于问题涉及多个子话题时。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户原始问题。",
            },
        },
        "required": ["query"],
    },
}

TOOL_HYDE_SEARCH = {
    "name": "hyde_search",
    "description": "用 HyDE（Hypothetical Document Embeddings）技术：先生成一段假想文档片段，用该片段的向量做检索。适用于 query 和文档用词差异大、cosine 低的情况。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户原始问题。",
            },
        },
        "required": ["query"],
    },
}

TOOL_RERANK = {
    "name": "rerank_results",
    "description": "用 Reranker 模型对检索结果进行精排，提升最相关片段在顶部的概率。适用于 concentration 低（多个文档各有一点相关）时。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户原始问题。",
            },
            "result_count": {
                "type": "integer",
                "description": "需要精排的结果数量，默认15。",
            },
        },
        "required": ["query"],
    },
}

TOOL_DECOMPOSE = {
    "name": "decompose_task",
    "description": "将复杂问题拆解为 2-5 个独立的子问题，每个子问题可独立检索后聚合。适用于需要综合多文档分析的复杂问题。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "需要拆解的复杂问题。",
            },
        },
        "required": ["query"],
    },
}

TOOL_ANALYZE_SIGNALS = {
    "name": "analyze_retrieval_signals",
    "description": "分析当前检索信号（cosine、concentration、info_gain），给出策略建议。不执行检索，仅做分析。",
    "parameters": {
        "type": "object",
        "properties": {
            "faiss_cosine": {
                "type": "number",
                "description": "FAISS top-1 cosine 分数",
            },
            "concentration": {
                "type": "number",
                "description": "文档集中度",
            },
            "info_gain": {
                "type": "number",
                "description": "最近一轮的信息增益（0-1）",
            },
            "hit_count": {
                "type": "integer",
                "description": "命中 chunk 数量",
            },
        },
        "required": ["faiss_cosine", "concentration"],
    },
}

TOOL_GENERATE_ANSWER = {
    "name": "generate_answer",
    "description": "基于检索到的文档片段生成最终回答。自动标注来源 [项目文档]/[通用知识]/[网络搜索]。",
    "parameters": {
        "type": "object",
        "properties": {
            "include_sections": {
                "type": "string",
                "description": "回答需要包含的部分，用逗号分隔。如 '概念解释,性能对比,风险分析'。",
            },
        },
        "required": [],
    },
}

TOOL_SEARCH_SCENARIO_KB = {
    "name": "search_scenario_kb",
    "description": "在运维场景知识库中检索结构化排障知识卡片。场景知识库包含 MySQL/Redis/K8s/Nginx/Kafka/ES/Docker/Java 等常见故障的排查步骤、根因分析和预防措施。当用户询问具体故障排查问题时优先使用此工具。结果标注 [知识库]。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询文本，应包含故障现象或技术关键词。",
            },
        },
        "required": ["query"],
    },
}

TOOL_WEB_SEARCH = {
    "name": "web_search",
    "description": "联网搜索，获取项目文档库未覆盖的外部信息。仅在文档库检索完全无结果或结果极差时使用。结果标注 [网络搜索]。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询。",
            },
        },
        "required": ["query"],
    },
}

TOOL_GENERATE_PPT = {
    "name": "generate_ppt",
    "description": "基于检索到的项目文档内容生成 PPT 演示文稿。适用于答辩材料、项目汇报、方案展示等场景。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "PPT 标题",
            },
            "outline": {
                "type": "string",
                "description": "PPT 大纲，用逗号分隔各页标题，如 '项目背景,技术架构,实施方案,风险分析,总结'",
            },
        },
        "required": ["title"],
    },
}

TOOL_GENERATE_WORD = {
    "name": "generate_word",
    "description": "基于检索到的项目文档内容生成 Word 文档。适用于技术方案、需求文档、项目总结等。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "文档标题",
            },
            "content_type": {
                "type": "string",
                "description": "文档类型: 技术方案 / 需求文档 / 项目总结 / 会议纪要",
            },
        },
        "required": ["title"],
    },
}

TOOL_GENERATE_PDF = {
    "name": "generate_pdf",
    "description": "基于检索到的项目文档内容生成 PDF 文档。适用于正式报告、归档文档等。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "PDF 文档标题",
            },
        },
        "required": ["title"],
    },
}

TOOL_GENERATE_EXCEL = {
    "name": "generate_excel",
    "description": "基于检索到的项目文档内容生成 Excel 表格。适用于数据汇总、对比分析、排期计划等。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "表格标题",
            },
            "sheet_type": {
                "type": "string",
                "description": "表格类型: 数据汇总 / 对比分析 / 排期计划 / 清单列表",
            },
        },
        "required": ["title"],
    },
}

TOOL_GENERATE_DIAGRAM = {
    "name": "generate_diagram",
    "description": "生成项目相关的图表（架构图、流程图、时序图等），输出 Mermaid 格式的图表代码。",
    "parameters": {
        "type": "object",
        "properties": {
            "diagram_type": {
                "type": "string",
                "description": "图表类型: architecture(架构图) / flowchart(流程图) / sequence(时序图) / class(类图)",
            },
            "description": {
                "type": "string",
                "description": "图表内容的文字描述",
            },
        },
        "required": ["diagram_type", "description"],
    },
}

TOOL_GENERATE_MARKDOWN = {
    "name": "generate_markdown",
    "description": "基于检索到的项目文档内容生成 Markdown 文档（.md 文件）。适用于技术文档、README、知识库文章等。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Markdown 文档标题",
            },
        },
        "required": ["title"],
    },
}


def _wrap_tool(tool_def: dict) -> dict:
    """将内部工具定义包装为 OpenAI function calling 格式。"""
    return {
        "type": "function",
        "function": tool_def,
    }


def get_all_tool_specs() -> list[dict]:
    """返回所有工具定义的列表（OpenAI function calling 格式）。"""
    tools = [
        _wrap_tool(TOOL_SEARCH_KNOWLEDGE_BASE),
        _wrap_tool(TOOL_SEARCH_SCENARIO_KB),
        _wrap_tool(TOOL_REWRITE_QUERY),
        _wrap_tool(TOOL_MULTI_QUERY),
        _wrap_tool(TOOL_HYDE_SEARCH),
        _wrap_tool(TOOL_RERANK),
        _wrap_tool(TOOL_DECOMPOSE),
        _wrap_tool(TOOL_ANALYZE_SIGNALS),
        _wrap_tool(TOOL_GENERATE_ANSWER),
        _wrap_tool(TOOL_WEB_SEARCH),
        _wrap_tool(TOOL_GENERATE_PPT),
        _wrap_tool(TOOL_GENERATE_WORD),
        _wrap_tool(TOOL_GENERATE_PDF),
        _wrap_tool(TOOL_GENERATE_EXCEL),
        _wrap_tool(TOOL_GENERATE_DIAGRAM),
        _wrap_tool(TOOL_GENERATE_MARKDOWN),
    ]
    from app.core.config import settings
    if not settings.scenario_kb_enabled:
        tools = [t for t in tools if t["function"]["name"] != "search_scenario_kb"]
    return tools


def get_tool_specs_for_class(classification) -> list[dict]:
    """根据问题分类返回可用的工具列表。

    shallow: 基础检索 + Query Rewrite + 生成回答
    deep:    全部工具
    """
    from app.agent.classifier import QueryClass
    from app.core.config import settings

    base_tools = [
        _wrap_tool(TOOL_SEARCH_KNOWLEDGE_BASE),
        _wrap_tool(TOOL_GENERATE_ANSWER),
    ]
    if settings.scenario_kb_enabled:
        base_tools.append(_wrap_tool(TOOL_SEARCH_SCENARIO_KB))

    if classification.query_class == QueryClass.CHITCHAT:
        return []  # 闲聊不需要工具

    if classification.query_class == QueryClass.SHALLOW:
        return base_tools + [
            _wrap_tool(TOOL_REWRITE_QUERY),
            _wrap_tool(TOOL_ANALYZE_SIGNALS),
            _wrap_tool(TOOL_WEB_SEARCH),
        ]

    # deep: 全部工具（含文档生成）
    return base_tools + [
        _wrap_tool(TOOL_REWRITE_QUERY),
        _wrap_tool(TOOL_MULTI_QUERY),
        _wrap_tool(TOOL_HYDE_SEARCH),
        _wrap_tool(TOOL_RERANK),
        _wrap_tool(TOOL_DECOMPOSE),
        _wrap_tool(TOOL_ANALYZE_SIGNALS),
        _wrap_tool(TOOL_WEB_SEARCH),
        _wrap_tool(TOOL_GENERATE_PPT),
        _wrap_tool(TOOL_GENERATE_WORD),
        _wrap_tool(TOOL_GENERATE_PDF),
        _wrap_tool(TOOL_GENERATE_EXCEL),
        _wrap_tool(TOOL_GENERATE_DIAGRAM),
        _wrap_tool(TOOL_GENERATE_MARKDOWN),
    ]


def get_tool_category(tool_name: str) -> str:
    """返回工具所属类别。"""
    mapping = {
        "search_knowledge_base": "search",
        "search_scenario_kb": "search",
        "rewrite_query": "search",
        "multi_query_search": "search",
        "hyde_search": "search",
        "rerank_results": "analysis",
        "decompose_task": "analysis",
        "analyze_retrieval_signals": "analysis",
        "generate_answer": "generation",
        "web_search": "mcp",
        "generate_ppt": "mcp",
        "generate_word": "mcp",
        "generate_pdf": "mcp",
        "generate_excel": "mcp",
        "generate_diagram": "mcp",
        "generate_markdown": "mcp",
    }
    return mapping.get(tool_name, "unknown")
