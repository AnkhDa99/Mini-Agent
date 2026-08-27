"""QualitySignal — 统一检索质量信号。

将散落在各处的质量阈值集中管理，所有决策点从这一个信号源取值。

两重评判 → 一个信号：
  第一重 concentration (纯数学, 0ms): 命中文档在语义空间的集中度
  第二重 completeness   (LLM评估): 检索内容能否充分回答用户问题

阈值常量集中定义，不在各处散落魔法数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ═════════════════════════════════════════════════════════════
# 阈值常量（唯一来源）
# ═════════════════════════════════════════════════════════════

# concentration 检索质量分档
CONCENTRATION_GOOD = 0.5       # ≥0.5 → good
CONCENTRATION_MARGINAL = 0.3   # ≥0.3 → marginal, <0.3 → poor

# completeness 完整性最低门槛
COMPLETENESS_MINIMUM = 40      # <40 → 结果不足以回答，触发降级

# info_gain 信息增益衰减
INFO_GAIN_MINIMUM = 0.2        # <0.2 → 本轮搜索几乎没带回新信息

# 终止计数器
CONSECUTIVE_ROUNDS_THRESHOLD = 2  # 连续 N 轮触发条件 → 终止

# 场景匹配
SCENARIO_HIGH_CONFIDENCE = 0.6    # ≥0.6 → 高置信，场景过滤启用
SCENARIO_STRONG_CONFIDENCE = 0.75  # ≥0.75 → 更严格的评分过滤

# 检索
SCENARIO_FAISS_THRESHOLD_DIRECT = 0.75  # 直接场景匹配
SCENARIO_FAISS_THRESHOLD_AGENT = 0.45   # Agent 主动检索时宽松阈值

# 降级链
DEGRADATION_LEVEL_CHITCHAT = 0
DEGRADATION_LEVEL_SHALLOW = 2
DEGRADATION_LEVEL_DEEP = 2

# Agent 循环上限
MAX_ROUNDS_CHITCHAT = 1
MAX_ROUNDS_SHALLOW = 2
MAX_ROUNDS_DEEP = 5


@dataclass
class QualitySignal:
    """统一检索质量信号，所有决策从这里取值。"""

    # ── 数学信号（无 LLM） ──
    concentration: float = 0.0          # 文档集中度 0-1
    faiss_ok: bool = False
    es_ok: bool = False
    faiss_conc: float = 0.0
    es_conc: float = 0.0
    info_gain: float = 0.0              # 信息增益 0-1
    scenario_confidence: float = 0.0    # 场景知识库置信度 0-1
    results_count: int = 0

    # ── LLM 评估信号 ──
    completeness: int = 0               # 完整度 0-100
    groundedness: int = 0               # 有据可依度 0-100
    uncertainties: list[str] = field(default_factory=list)
    decision: str = ""                  # CONTINUE | STOP
    decision_reason: str = ""
    uncertainty_handling: str = "retry" # retry | admit | web_search | general_knowledge

    # ── 派生属性（阈值集中管理） ──

    @property
    def retrieval_grade(self) -> str:
        """检索质量等级：good | marginal | poor | empty"""
        if not self.faiss_ok and not self.es_ok:
            return "empty"
        if self.concentration >= CONCENTRATION_GOOD:
            return "good"
        if self.concentration >= CONCENTRATION_MARGINAL:
            return "marginal"
        return "poor"

    @property
    def is_empty(self) -> bool:
        return self.retrieval_grade == "empty"

    @property
    def is_poor(self) -> bool:
        return self.retrieval_grade == "poor"

    @property
    def is_marginal(self) -> bool:
        return self.retrieval_grade == "marginal"

    @property
    def is_good(self) -> bool:
        return self.retrieval_grade == "good"

    @property
    def should_degrade_by_concentration(self) -> bool:
        """仅凭 concentration 判断是否需要降级。"""
        return self.retrieval_grade in ("empty", "poor")

    @property
    def should_degrade_by_completeness(self) -> bool:
        """completeness 不足 → 需要降级。"""
        return self.completeness < COMPLETENESS_MINIMUM and self.completeness > 0

    @property
    def should_degrade(self) -> bool:
        """综合判断：是否应触发降级。"""
        return self.should_degrade_by_concentration or self.should_degrade_by_completeness

    @property
    def should_terminate_low_gain(self) -> bool:
        """本轮信息增益不足。"""
        return self.info_gain < INFO_GAIN_MINIMUM

    @property
    def scenario_is_high_confidence(self) -> bool:
        return self.scenario_confidence >= SCENARIO_HIGH_CONFIDENCE

    @property
    def scenario_is_strong_confidence(self) -> bool:
        return self.scenario_confidence >= SCENARIO_STRONG_CONFIDENCE

    # ── 工厂方法 ──

    @classmethod
    def from_fusion(cls, fusion_info: dict | None, results_count: int = 0) -> "QualitySignal":
        """从融合信息创建初始信号（仅数学部分，LLM 部分后续填充）。"""
        if not fusion_info:
            return cls(results_count=results_count)

        faiss_ok = fusion_info.get("faiss_ok", False)
        es_ok = fusion_info.get("es_ok", False)
        faiss_conc = fusion_info.get("faiss_conc", 0)
        es_conc = fusion_info.get("es_conc", 0)
        concentration = max(faiss_conc, es_conc)

        return cls(
            concentration=concentration,
            faiss_ok=faiss_ok,
            es_ok=es_ok,
            faiss_conc=faiss_conc,
            es_conc=es_conc,
            results_count=results_count,
        )

    def merge_llm_assessment(
        self,
        completeness: int,
        groundedness: int,
        uncertainties: list[str] | None = None,
        decision: str = "",
        decision_reason: str = "",
        uncertainty_handling: str = "retry",
    ) -> "QualitySignal":
        """合并 LLM 评估结果到信号中。"""
        self.completeness = completeness
        self.groundedness = groundedness
        if uncertainties:
            self.uncertainties = list(uncertainties)
        self.decision = decision
        self.decision_reason = decision_reason
        self.uncertainty_handling = uncertainty_handling
        return self

    def to_dict(self) -> dict:
        return {
            "retrieval_grade": self.retrieval_grade,
            "concentration": self.concentration,
            "completeness": self.completeness,
            "groundedness": self.groundedness,
            "info_gain": self.info_gain,
            "scenario_confidence": self.scenario_confidence,
            "decision": self.decision,
            "should_degrade": self.should_degrade,
        }
