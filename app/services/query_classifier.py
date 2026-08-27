"""
Query 复杂度分类器：识别简单/事实/分析/复杂问题，决定检索策略路由。

采用 A+B 混合方案：
- 正则快速过滤明显问题
- 模糊地带由 LLM 判断

为 Agent 预留接口：Agent 可直接调用 classify() 获取分类结果，
并通过 classification_hook 注入自定义分类逻辑。
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    SIMPLE = "simple"          # 简短关键词/寒暄 → ES-only, 无扩展, 无精排
    FACTUAL = "factual"        # 事实查询/定义 → ES+FAISS, 无扩展, 无精排
    ANALYTICAL = "analytical"  # 需要综合分析 → 完整检索, 无 Plan-Execute
    COMPLEX = "complex"        # 多步推理/任务拆解 → Plan-and-Execute + Reranker


@dataclass
class ClassificationResult:
    complexity: QueryComplexity
    confidence: float          # 0.0-1.0
    reason: str                # 分类依据
    needs_expansion: bool = False
    needs_rerank: bool = False
    needs_plan_execute: bool = False


# ── 正则规则 ──

SIMPLE_PATTERNS: list[tuple[str, QueryComplexity]] = [
    # 寒暄 / 闲聊
    (r'^(你好|谢谢|再见|在吗|hello|hi|ok|好的|是的|对的|嗯|哦|哈哈|好的谢谢)$', QueryComplexity.SIMPLE),
    # 单个词或短术语（无上下文）
    (r'^[\w一-鿿]{1,6}$', QueryComplexity.SIMPLE),
]

FACTUAL_PATTERNS: list[tuple[str, QueryComplexity]] = [
    # 定义类
    (r'^(什么是|什么叫|定义|解释)[一-鿿\w]+', QueryComplexity.FACTUAL),
    (r'^[一-鿿\w]+(是什么|指的是|是指|的定义)', QueryComplexity.FACTUAL),
    # 简单参数/配置查询
    (r'^(如何设置|怎么配置|参数|默认值|端口|地址).{0,20}$', QueryComplexity.FACTUAL),
    # 列举类
    (r'^(有哪些|列出|列举|几个|哪些).{0,15}$', QueryComplexity.FACTUAL),
]

COMPLEX_PATTERNS: list[tuple[str, QueryComplexity]] = [
    # 需求/设计/分析
    (r'(需求分析|方案设计|架构设计|系统设计|技术选型)', QueryComplexity.COMPLEX),
    # 对比+多个维度
    (r'.{5,}(区别|对比|比较|优缺点).{5,}(区别|对比|比较|优缺点)', QueryComplexity.COMPLEX),
    # 风险/优化/评估
    (r'.{10,}(风险评估|性能优化|优化建议|如何改进|改进方案)', QueryComplexity.COMPLEX),
    # 多步骤任务
    (r'.{10,}(如何实现|怎么搭建|怎么部署|如何部署|从零|cicd|pipeline)', QueryComplexity.COMPLEX),
    # 周报/答辩/面试材料
    (r'(周报|日报|答辩|面试材料|项目总结|月度总结)', QueryComplexity.COMPLEX),
]

ANALYTICAL_PATTERNS: list[tuple[str, QueryComplexity]] = [
    # 比较/对比（单维度）
    (r'.{5,}(区别|对比|比较|优缺点|哪个更好).{3,}', QueryComplexity.ANALYTICAL),
    # 原理/机制
    (r'.{5,}(原理|机制|流程|过程|怎么工作|如何运作)', QueryComplexity.ANALYTICAL),
    # 原因/为什么
    (r'.{5,}(为什么|原因是|什么原因|怎么会)', QueryComplexity.ANALYTICAL),
    # 如何做（中等复杂度）
    (r'(如何|怎么|怎样).{5,}', QueryComplexity.ANALYTICAL),
]


class QueryClassifier:
    """查询复杂度分类器。

    为 Agent 预留的扩展点：
    - classification_hook: Agent 可注入自定义分类函数
    - classify() 返回结构化结果，Agent 可直接消费
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client
        # Agent 可注入自定义分类逻辑
        self.classification_hook: Callable[[str], ClassificationResult | None] | None = None

    def classify(self, query: str) -> ClassificationResult:
        """分类主入口。先正则，模糊用 LLM，Agent hook 可覆盖。"""
        query_stripped = query.strip()

        # ── Agent hook 优先 ──
        if self.classification_hook:
            result = self.classification_hook(query_stripped)
            if result is not None:
                return result

        # ── 正则快速判断 ──
        for pattern, complexity in SIMPLE_PATTERNS:
            if re.match(pattern, query_stripped, re.IGNORECASE):
                if complexity == QueryComplexity.SIMPLE:
                    return ClassificationResult(
                        complexity=QueryComplexity.SIMPLE,
                        confidence=1.0,
                        reason=f"regex match: simple pattern",
                    )

        for pattern, complexity in FACTUAL_PATTERNS:
            if re.match(pattern, query_stripped, re.IGNORECASE):
                return ClassificationResult(
                    complexity=QueryComplexity.FACTUAL,
                    confidence=0.85,
                    reason=f"regex match: factual pattern",
                )

        for pattern, complexity in COMPLEX_PATTERNS:
            if re.match(pattern, query_stripped, re.IGNORECASE):
                return ClassificationResult(
                    complexity=QueryComplexity.COMPLEX,
                    confidence=0.8,
                    reason=f"regex match: complex pattern",
                    needs_expansion=True,
                    needs_rerank=True,
                    needs_plan_execute=True,
                )

        for pattern, complexity in ANALYTICAL_PATTERNS:
            if re.match(pattern, query_stripped, re.IGNORECASE):
                return ClassificationResult(
                    complexity=QueryComplexity.ANALYTICAL,
                    confidence=0.75,
                    reason=f"regex match: analytical pattern",
                    needs_expansion=True,
                )

        # ── 模糊地带：长度 + 关键词启发式 ──
        if len(query_stripped) < 10:
            return ClassificationResult(
                complexity=QueryComplexity.FACTUAL,
                confidence=0.5,
                reason="short query, default to factual",
            )

        # ── LLM 细粒度分类 ──
        if self._llm:
            return self._llm_classify(query_stripped)

        # 兜底
        return ClassificationResult(
            complexity=QueryComplexity.ANALYTICAL,
            confidence=0.4,
            reason="fallback: no LLM available, default analytical",
        )

    def _llm_classify(self, query: str) -> ClassificationResult:
        """用 LLM 做细粒度分类。"""
        prompt = (
            "评估以下问题的复杂度类型（只输出一个词）:\n"
            "simple=简短寒暄/单关键词 | factual=事实查询/定义/参数 | "
            "analytical=需要分析/对比/推理 | complex=多步任务/方案设计/需求分析\n"
            f"问题: {query}\n"
            "类型:"
        )
        try:
            resp = self._llm.chat([{"role": "user", "content": prompt}])
            resp = resp.strip().lower()
            mapping = {
                "simple": (QueryComplexity.SIMPLE, 0.7),
                "factual": (QueryComplexity.FACTUAL, 0.7),
                "analytical": (QueryComplexity.ANALYTICAL, 0.7),
                "complex": (QueryComplexity.COMPLEX, 0.7),
            }
            for key, (comp, conf) in mapping.items():
                if key in resp:
                    return ClassificationResult(
                        complexity=comp,
                        confidence=conf,
                        reason=f"LLM classified as {key}",
                        needs_expansion=comp in (QueryComplexity.ANALYTICAL, QueryComplexity.COMPLEX),
                        needs_rerank=comp == QueryComplexity.COMPLEX,
                        needs_plan_execute=comp == QueryComplexity.COMPLEX,
                    )
        except Exception:
            logger.exception("LLM classification failed, fallback")

        return ClassificationResult(
            complexity=QueryComplexity.ANALYTICAL,
            confidence=0.3,
            reason="LLM classification failed, fallback analytical",
        )


# 全局单例，支持惰性注入 LLM client
_classifier: QueryClassifier | None = None


def get_classifier(llm_client=None) -> QueryClassifier:
    global _classifier
    if _classifier is None or (llm_client and _classifier._llm is None):
        _classifier = QueryClassifier(llm_client)
    return _classifier
