"""
RRF 自适应权重：根据 FAISS/ES 各自检索信号强度动态调整融合权重。

原理：
- FAISS top1 远高于其余 → 向量信号强 → 加重 FAISS
- ES 分数标准差大 → 关键词信号强 → 加重 ES
- 双路重叠度高 → 信号一致 → 等权

为 Agent 预留接口：
- Agent 可通过 weight_override 注入自定义权重策略
"""
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

RRF_K = 60
DEFAULT_W_FAISS = 0.5
DEFAULT_W_ES = 0.5


@dataclass
class RRFWeights:
    w_faiss: float
    w_es: float
    faiss_signal: float       # 0-1, FAISS 信号强度
    es_signal: float          # 0-1, ES 信号强度
    overlap: float            # 双路重叠度
    reason: str


class AdaptiveRRF:
    """自适应 RRF 融合器。

    Agent 扩展点:
    - weight_override_hook(faiss_scores, es_scores, overlap) → RRFWeights | None
    """

    def __init__(self):
        self.weight_override_hook: Callable[[list[float], list[float], float], RRFWeights | None] | None = None

    def compute_weights(
        self, faiss_scores: list[float], es_scores: list[float], overlap: float
    ) -> RRFWeights:
        """计算 FAISS/ES 的自适应权重。"""
        # ── Agent 自定义权重优先 ──
        if self.weight_override_hook:
            result = self.weight_override_hook(faiss_scores, es_scores, overlap)
            if result is not None:
                return result

        n_faiss = len(faiss_scores)
        n_es = len(es_scores)

        # FAISS 信号强度：top1 与均值之比 + 分数衰减速度
        if n_faiss >= 2 and faiss_scores[0] > 0:
            faiss_top1 = faiss_scores[0]
            faiss_mean = sum(faiss_scores) / n_faiss
            # top1 越突出，信号越强
            faiss_peak = min(1.0, faiss_top1 / max(faiss_mean, 0.01))
            # 衰减越快（top1 vs top5），信号越强
            if n_faiss >= 5:
                faiss_decay = min(1.0, 1.0 - (faiss_scores[4] / max(faiss_top1, 0.01)))
            else:
                faiss_decay = 0.5
            faiss_signal = 0.5 * faiss_peak + 0.5 * faiss_decay
        else:
            faiss_signal = 0.1

        # ES 信号强度：top1 绝对值 + 分数标准差
        if n_es >= 2 and es_scores[0] > 0:
            es_top1 = es_scores[0]
            # BM25 分数 > 10 表示高相关性
            es_abs = min(1.0, es_top1 / 15.0)
            es_mean = sum(es_scores) / n_es
            max_score = max(es_scores)
            es_std = (sum((s - es_mean) ** 2 for s in es_scores) / n_es) ** 0.5
            es_cv = min(1.0, es_std / max(max_score, 0.01))
            es_signal = 0.5 * es_abs + 0.5 * es_cv
        else:
            es_signal = 0.1

        # 组合权重：信号强度决定偏离等权的幅度
        signal_diff = faiss_signal - es_signal
        adjustment = 0.3 * signal_diff  # 最大偏移 ±0.3
        w_faiss = max(0.1, min(0.9, DEFAULT_W_FAISS + adjustment))
        w_es = 1.0 - w_faiss

        # 重叠度高时倾向于等权
        if overlap > 0.5:
            w_faiss = w_faiss * 0.5 + DEFAULT_W_FAISS * 0.5
            w_es = 1.0 - w_faiss

        reason = (
            f"faiss_signal={faiss_signal:.2f} es_signal={es_signal:.2f} "
            f"overlap={overlap:.1%} → w_faiss={w_faiss:.2f} w_es={w_es:.2f}"
        )
        logger.debug("RRF adaptive: %s", reason)

        return RRFWeights(
            w_faiss=round(w_faiss, 3),
            w_es=round(w_es, 3),
            faiss_signal=round(faiss_signal, 3),
            es_signal=round(es_signal, 3),
            overlap=round(overlap, 3),
            reason=reason,
        )

    def rrf_score(
        self, faiss_rank: int | None, es_rank: int | None, weights: RRFWeights
    ) -> float:
        """计算加权 RRF 分数。"""
        score = 0.0
        if faiss_rank is not None and faiss_rank >= 0 and weights.w_faiss > 0:
            score += weights.w_faiss / (RRF_K + faiss_rank + 1)
        if es_rank is not None and es_rank >= 0 and weights.w_es > 0:
            score += weights.w_es / (RRF_K + es_rank + 1)
        return score


# 全局单例
_adaptive_rrf: AdaptiveRRF | None = None


def get_adaptive_rrf() -> AdaptiveRRF:
    global _adaptive_rrf
    if _adaptive_rrf is None:
        _adaptive_rrf = AdaptiveRRF()
    return _adaptive_rrf
