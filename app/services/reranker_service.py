"""
Cross-Encoder Reranker: 对 RRF 融合后的候选做精排。
使用 BAAI/bge-reranker-v2-m3（多语言，中英文均优）。
加载失败自动降级，不影响主流程。
"""
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# 精排时从 RRF merged 候选中取多少条喂给 reranker
RERANK_CANDIDATE_K = 30
# 精排后返回多少条
RERANK_TOP_K = 10


class RerankerService:
    """Cross-encoder 精排器。lazy-load 模型，加载失败直接透传。"""

    def __init__(self):
        self._model = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        if not settings.reranker_enabled:
            return False
        if self._load_failed:
            return False
        return self._ensure_model()

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import os

            from sentence_transformers import CrossEncoder

            # HuggingFace 镜像（国内网络加速），本地路径则需禁用在线检查
            if settings.hf_endpoint:
                os.environ["HF_ENDPOINT"] = settings.hf_endpoint
            # 如果配置指向本地已存在的目录，直接用离线模式，不连 HF
            if os.path.isdir(settings.reranker_model):
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            self._model = CrossEncoder(
                settings.reranker_model,
                max_length=512,
                trust_remote_code=True,
            )
            # warmup
            _ = self._model.predict([("test query", "test passage")])
            logger.info("Reranker loaded | model=%s", settings.reranker_model)
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, reranker disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._load_failed = True
            return False
        except Exception as e:
            logger.warning("Reranker model load failed: %s, reranker disabled", e)
            self._load_failed = True
            return False

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[dict]:
        """
        对 RRF 融合后的候选做 cross-encoder 精排。
        返回重新排序后的 top_k 结果。
        """
        if not candidates or not self.available:
            return candidates[:top_k]

        # 只精排前 RERANK_CANDIDATE_K 条（RRF 粗排已过滤噪声）
        pool = candidates[:RERANK_CANDIDATE_K]

        pairs = [(query, c.get("content", "")[:512]) for c in pool]
        try:
            scores = self._model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("Reranker inference failed, falling back to RRF order")
            return candidates[:top_k]

        for i, c in enumerate(pool):
            c["rerank_score"] = round(float(scores[i]), 4)

        ranked = sorted(pool, key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(
            "Reranker | candidates=%d → top=%d | "
            "top1_rerank=%.4f rrf=%.4f file=%s",
            len(pool), min(top_k, len(ranked)),
            ranked[0].get("rerank_score", 0) if ranked else 0,
            ranked[0].get("rrf_score", 0) if ranked else 0,
            ranked[0].get("filename", "")[:30] if ranked else "",
        )
        return ranked[:top_k]
