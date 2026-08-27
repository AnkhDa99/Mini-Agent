"""
SearchRouter — Workflow 级联检索路由。

根据 QueryClassifier 的分类结果，将查询路由到对应的检索流水线：

  SIMPLE    → ES-only (10ms)
  FACTUAL   → ES + FAISS + RRF (200ms)
  ANALYTICAL → ES + FAISS + Multi-Query + HyDE + RRF (500ms)
  COMPLEX   → Plan-and-Execute (拆解→多路检索→聚合→Reranker) (2-5s)

集成 EmbeddingCache / SearchCache / AdaptiveRRF。

为 Agent 预留接口：
- SearchRouter.register_strategy(name, strategy) → Agent 注册自定义检索策略
- SearchRouter.set_planner(planner) → Agent 注入任务规划器
- SearchRouter.override_route(query) → Agent 覆盖路由决策
"""
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.embedding_cache import get_embedding_cache
from app.services.embedding_service import EmbeddingService
from app.services.es_service import ESService
from app.services.hybrid_search import HybridSearchService
from app.services.query_classifier import (
    ClassificationResult,
    QueryClassifier,
    QueryComplexity,
    get_classifier,
)
from app.services.reranker_service import RerankerService
from app.services.rrf_adaptive import AdaptiveRRF, RRFWeights, get_adaptive_rrf
from app.services.search_cache import get_search_cache
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

RETRIEVAL_K = 25
TOP_K = 10
RRF_K = 60
RERANK_CANDIDATE_K = 30


@dataclass
class SearchContext:
    """检索上下文，在流水线各阶段传递。Agent 可读写。"""
    query: str
    classification: ClassificationResult
    # 中间结果
    faiss_results: list[dict] = field(default_factory=list)
    es_results: list[dict] = field(default_factory=list)
    merged_results: list[dict] = field(default_factory=list)
    fusion_info: dict | None = field(default=None)
    # 耗时统计
    timings: dict[str, float] = field(default_factory=dict)
    # 是否命中缓存
    cache_hit: bool = False


class SearchRouter:
    """Workflow 级联检索路由器。

    Agent 扩展点:
    - register_strategy(name, fn): 注册自定义检索策略
    - set_planner(planner): 注入任务规划器（用于 COMPLEX 路径的拆解）
    - override_route_hook(ctx) → SearchContext | None: 覆盖路由决策
    - post_search_hook(ctx) → SearchContext: 检索后处理
    """

    def __init__(self):
        self._embedding: EmbeddingService | None = None
        self._vector_store: VectorStore | None = None
        self._es: ESService | None = None
        self._reranker: RerankerService | None = None
        self._classifier: QueryClassifier | None = None
        self._adaptive_rrf: AdaptiveRRF | None = None

        # Agent 扩展点
        self._strategies: dict[str, Callable] = {}
        self._planner: Any = None
        self.override_route_hook: Callable[[SearchContext], SearchContext | None] | None = None
        self.post_search_hook: Callable[[SearchContext], SearchContext] | None = None

    # ── 懒加载属性 ──

    @property
    def embedding(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def es(self) -> ESService | None:
        if self._es is None:
            try:
                self._es = ESService()
            except Exception:
                logger.warning("ES unavailable")
        return self._es

    @property
    def reranker(self) -> RerankerService:
        if self._reranker is None:
            self._reranker = RerankerService()
        return self._reranker

    @property
    def classifier(self) -> QueryClassifier:
        if self._classifier is None:
            from app.llm.openai_client import OpenAIChatClient
            try:
                llm = OpenAIChatClient()
            except Exception:
                llm = None
            self._classifier = QueryClassifier(llm)
        return self._classifier

    @property
    def adaptive_rrf(self) -> AdaptiveRRF:
        if self._adaptive_rrf is None:
            self._adaptive_rrf = get_adaptive_rrf()
        return self._adaptive_rrf

    # ── Agent 接口 ──

    def register_strategy(self, name: str, strategy_fn: Callable):
        """Agent 注册自定义检索策略。"""
        self._strategies[name] = strategy_fn
        logger.info("Agent registered search strategy: %s", name)

    def set_planner(self, planner):
        """Agent 注入任务规划器。planner.plan(query) → list[SubTask]"""
        self._planner = planner
        logger.info("Agent planner injected: %s", type(planner).__name__)

    # ── 主入口 ──

    def search(
        self, query: str, db, *, classify_query: str = "", _depth: int = 0,
        timeout_ms: float = 60000.0, owner_id: int | None = None, **kwargs
    ) -> tuple[list[dict], SearchContext]:
        """Workflow 检索主入口。返回 (results, context)。

        query: 用于检索的查询文本（可能是上下文增强后的）
        classify_query: 用于分类的原始查询（不传则等于 query），避免上下文拼接干扰正则匹配
        _depth: 内部递归深度（0=顶层），Complex 子查询传 _depth=1 防止无限递归。
        timeout_ms: 整体超时（毫秒），超时后返回已有结果不再扩展。
        owner_id: 数据隔离过滤（非 None 则仅返回该用户的 chunks）。
        """
        t0 = time.perf_counter()
        deadline = t0 + timeout_ms / 1000.0

        def _timed_out() -> bool:
            return time.perf_counter() > deadline

        ctx = SearchContext(query=query, classification=ClassificationResult(
            complexity=QueryComplexity.ANALYTICAL,
            confidence=0.0,
            reason="initial",
        ))

        # ── 尝试检索结果缓存 ──
        search_cache = get_search_cache()
        if search_cache.available:
            cached_uids = search_cache.get(query, owner_id=owner_id)
            if cached_uids:
                ctx.cache_hit = True
                ctx.timings["cache"] = (time.perf_counter() - t0) * 1000
                logger.info("SearchRouter cache HIT | q=%.40s", query)
                results = self._load_cached_results(db, cached_uids, owner_id=owner_id)
                if results:
                    ctx.timings["total"] = (time.perf_counter() - t0) * 1000
                    return results, ctx

        # ── 分类（子查询跳过 LLM 分类，直接走 factual）──
        # 用原始查询做分类，避免上下文拼接（如"对话上文:..."）干扰正则匹配
        classify_text = classify_query or query
        if _depth >= 1:
            ctx.classification = ClassificationResult(
                complexity=QueryComplexity.FACTUAL,
                confidence=0.8,
                reason=f"sub-query depth={_depth}",
            )
        else:
            ctx.classification = self.classifier.classify(classify_text)
        ctx.timings["classify"] = (time.perf_counter() - t0) * 1000

        # ── Agent 可覆盖路由 ──
        if self.override_route_hook:
            overridden = self.override_route_hook(ctx)
            if overridden is not None:
                ctx = overridden

        logger.info(
            "SearchRouter | q=%.50s | complexity=%s conf=%.2f depth=%d reason=%s",
            query, ctx.classification.complexity.value,
            ctx.classification.confidence, _depth, ctx.classification.reason,
        )

        # ── 路由 ──
        complexity = ctx.classification.complexity
        # 深层递归强制降级: COMPLEX → ANALYTICAL，避免无限展开
        if _depth >= 1 and complexity == QueryComplexity.COMPLEX:
            complexity = QueryComplexity.ANALYTICAL
            ctx.classification.complexity = complexity
            ctx.classification.reason = "depth-guard: complex→analytical"

        if complexity == QueryComplexity.SIMPLE:
            results = self._route_simple(ctx, db, owner_id=owner_id)
        elif complexity == QueryComplexity.FACTUAL:
            results = self._route_factual(ctx, db, deadline=deadline, owner_id=owner_id)
        elif complexity == QueryComplexity.COMPLEX:
            results = self._route_complex(ctx, db, deadline=deadline, owner_id=owner_id)
        else:  # ANALYTICAL
            results = self._route_analytical(ctx, db, deadline=deadline, owner_id=owner_id)

        # ── 第二层检索: DocSummaryIndex ──
        if _depth == 0 and not _timed_out():
            results = self._boost_by_doc_summary(ctx.query, results, db)

        ctx.timings["total"] = (time.perf_counter() - t0) * 1000

        # ── 计时摘要 ──
        _log_timing_summary(ctx)

        # ── Agent 后处理 ──
        if self.post_search_hook:
            ctx = self.post_search_hook(ctx)

        # ── 缓存结果（统一 strategy="default" 确保 get/set key 一致）──
        if search_cache.available and not ctx.cache_hit and results:
            chunk_uids = [r.get("chunk_uid", "") for r in results]
            search_cache.set(query, chunk_uids, strategy="default", owner_id=owner_id)

        return results, ctx

    # ── 路由实现 ──

    def _route_simple(self, ctx: SearchContext, db, owner_id: int | None = None) -> list[dict]:
        """ES-only 快速路径。"""
        t0 = time.perf_counter()
        if self.es:
            raw = self.es.search(ctx.query, k=TOP_K)
            ctx.es_results = raw
        ctx.timings["es"] = (time.perf_counter() - t0) * 1000
        return self._enrich_results(db, ctx.es_results, owner_id=owner_id)

    def _route_factual(self, ctx: SearchContext, db, deadline: float = 0, owner_id: int | None = None) -> list[dict]:
        """ES + FAISS + 自适应 RRF，无扩展，无精排。"""
        t0 = time.perf_counter()

        # FAISS
        query_vec = self._embed_with_cache(ctx.query)
        faiss_raw = self.vector_store.search(query_vec, k=RETRIEVAL_K)
        ctx.timings["faiss"] = (time.perf_counter() - t0) * 1000
        t_es = time.perf_counter()

        # ES
        es_raw: list[dict] = []
        if self.es:
            es_raw = self.es.search(ctx.query, k=RETRIEVAL_K)
        ctx.timings["es"] = (time.perf_counter() - t_es) * 1000

        # 加载 chunk（owner_id 过滤实现数据隔离）
        all_uids = {cuid for cuid, _ in faiss_raw} | {h.get("chunk_uid", "") for h in es_raw}
        chunk_map = self._load_chunks(db, list(all_uids), owner_id=owner_id)

        # Concentration-RRF 融合
        results, fusion_info = self._concentration_rrf_fusion(faiss_raw, es_raw, chunk_map)
        ctx.fusion_info = fusion_info
        ctx.merged_results = results
        ctx.timings["fusion"] = (time.perf_counter() - t0) * 1000 - ctx.timings.get("faiss", 0) - ctx.timings.get("es", 0)

        return results

    def _route_analytical(self, ctx: SearchContext, db, deadline: float = 0, owner_id: int | None = None) -> list[dict]:
        """完整检索 + Multi-Query + HyDE 扩展。若 LLM 超时则降级为 factual。"""
        from app.llm.openai_client import OpenAIChatClient
        from app.prompts.system_prompts import HYDE_PROMPT, MULTI_QUERY_PROMPT

        _timed_out = (deadline > 0 and time.perf_counter() > deadline)
        if _timed_out:
            logger.warning("Analytical: timed out before LLM, falling back to factual")
            return self._route_factual(ctx, db, deadline, owner_id=owner_id)

        llm = OpenAIChatClient()

        # Multi-Query 扩展（带超时保护）
        variants = self._expand_queries_llm(llm, ctx.query)
        all_queries = [ctx.query] + variants

        # 多路检索
        all_merged: dict[str, dict] = {}
        for round_idx, q in enumerate(all_queries[:3]):  # 最多 3 轮（避免过度扩展）
            if deadline > 0 and time.perf_counter() > deadline:
                logger.warning("Analytical: timed out at round %d, using partial results", round_idx)
                break

            # HyDE（仅第一轮，带超时保护）
            hyde_q = ""
            if round_idx == 0:
                try:
                    hyde_msg = [{"role": "user", "content": HYDE_PROMPT.format(query=q)}]
                    hyde_q = llm.chat(hyde_msg).strip()
                except Exception:
                    logger.debug("HyDE failed for q=%.40s, using original", q)

            query_vec = self._embed_with_cache(hyde_q or q)
            faiss_raw = self.vector_store.search(query_vec, k=RETRIEVAL_K)

            es_raw: list[dict] = []
            if self.es:
                es_raw = self.es.search(q, k=RETRIEVAL_K)

            all_uids = {cuid for cuid, _ in faiss_raw} | {h.get("chunk_uid", "") for h in es_raw}
            chunk_map = self._load_chunks(db, list(all_uids), owner_id=owner_id)

            # Score-based 融合 per round
            round_results, round_fusion = self._concentration_rrf_fusion(faiss_raw, es_raw, chunk_map)
            if round_idx == 0:
                ctx.fusion_info = round_fusion

            weight = 1.0 if round_idx == 0 else 0.6
            for r in round_results:
                cuid = r.get("chunk_uid", "")
                if cuid not in all_merged:
                    all_merged[cuid] = r
                    all_merged[cuid]["rrf_score"] = r.get("rrf_score", 0) * weight
                else:
                    all_merged[cuid]["rrf_score"] = max(
                        all_merged[cuid].get("rrf_score", 0),
                        r.get("rrf_score", 0) * weight,
                    )
                    all_merged[cuid]["faiss_score"] = max(
                        all_merged[cuid].get("faiss_score", 0), r.get("faiss_score", 0),
                    )
                    all_merged[cuid]["es_score"] = max(
                        all_merged[cuid].get("es_score", 0), r.get("es_score", 0),
                    )

        ranked = sorted(all_merged.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        ctx.merged_results = ranked

        # Reranker（如果可用且复杂度足够）—— 跳过以避免 CPU 开销，仅 Complex 路径触发
        if ctx.classification.needs_rerank and self.reranker.available and not _timed_out:
            try:
                pool = ranked[:RERANK_CANDIDATE_K]
                ranked = self.reranker.rerank(ctx.query, pool, top_k=TOP_K)
            except Exception:
                logger.exception("Reranker failed in analytical path")

        return ranked[:TOP_K]

    def _route_complex(self, ctx: SearchContext, db, deadline: float = 0, owner_id: int | None = None) -> list[dict]:
        """Plan-and-Execute: LLM 拆解 → 子问题直接调用 _route_factual → 聚合 → Reranker。

        关键修复:
        - 子查询走 _route_factual（跳过 LLM 分类 + 路由），不再递归 self.search()
        - 每个子查询有独立超时（10s）
        - 整体 deadline 控制
        - 拆解失败/超时 → 降级 analytical
        - sub_queries 上限 3 个（避免过度扩展）
        """
        from app.llm.openai_client import OpenAIChatClient

        t0 = time.perf_counter()
        _timed_out = (deadline > 0 and time.perf_counter() > deadline)

        # 已经超时，直接走快速路径
        if _timed_out:
            logger.warning("Complex: timed out before LLM plan, falling back to factual")
            return self._route_factual(ctx, db, deadline, owner_id=owner_id)

        llm = OpenAIChatClient()

        # Step 1: Plan — LLM 拆解（有超时保护）
        sub_queries = self._plan_decompose(llm, ctx.query)
        ctx.timings["plan_llm"] = (time.perf_counter() - t0) * 1000

        if not sub_queries:
            logger.warning("Plan decomposition failed for: %.50s, falling back to analytical", ctx.query)
            ctx.classification.needs_rerank = True
            return self._route_analytical(ctx, db, deadline, owner_id=owner_id)

        # 最多 3 个子问题（避免递归爆炸）
        sub_queries = sub_queries[:3]
        logger.info("Plan decomposed: %.50s → %d sub-queries", ctx.query, len(sub_queries))

        # Step 2: Execute — 子问题使用 SUB_QUERY_TIMEOUT_MS 直接走 factual
        SUB_QUERY_TIMEOUT_MS = 10000  # 子查询硬超时 10s
        all_merged: dict[str, dict] = {}
        sub_timings: list[float] = []

        for i, sq in enumerate(sub_queries):
            # 检查整体 deadline
            if deadline > 0 and time.perf_counter() > deadline:
                logger.warning("Complex: global timeout at sub-query %d/%d, using %d results",
                               i, len(sub_queries), len(all_merged))
                break

            t_sub = time.perf_counter()
            try:
                sub_ctx = SearchContext(
                    query=sq,
                    classification=ClassificationResult(
                        complexity=QueryComplexity.FACTUAL,
                        confidence=0.9,
                        reason=f"plan-sub-{i}",
                    ),
                )
                sub_deadline = time.perf_counter() + SUB_QUERY_TIMEOUT_MS / 1000.0
                sub_results = self._route_factual(sub_ctx, db, deadline=sub_deadline, owner_id=owner_id)

                # 聚合：取最高 rrf_score
                for r in sub_results:
                    cuid = r.get("chunk_uid", "")
                    if cuid not in all_merged:
                        all_merged[cuid] = r
                        all_merged[cuid]["rrf_score"] = r.get("rrf_score", 0)
                    else:
                        all_merged[cuid]["rrf_score"] = max(
                            all_merged[cuid].get("rrf_score", 0),
                            r.get("rrf_score", 0),
                        )
                sub_timings.append((time.perf_counter() - t_sub) * 1000)
            except Exception:
                logger.exception("Complex sub-query %d failed: %.40s", i, sq)
                sub_timings.append((time.perf_counter() - t_sub) * 1000)

        ctx.timings["sub_queries"] = sum(sub_timings)
        ctx.timings["sub_queries_detail"] = sub_timings

        ranked = sorted(all_merged.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        ctx.merged_results = ranked
        ctx.timings["plan_execute"] = (time.perf_counter() - t0) * 1000

        # Step 3: Reranker 精排（仅长结果集触发）
        if self.reranker.available and len(ranked) > TOP_K:
            if deadline > 0 and time.perf_counter() > deadline:
                logger.debug("Complex: skip reranker due to timeout")
            else:
                try:
                    pool = ranked[:RERANK_CANDIDATE_K]
                    ranked = self.reranker.rerank(ctx.query, pool, top_k=TOP_K)
                except Exception:
                    logger.exception("Reranker failed in complex path")

        return ranked[:TOP_K]

    # ── 辅助方法 ──

    def _embed_with_cache(self, query: str) -> list[float]:
        """带缓存的 Embedding。"""
        emb_cache = get_embedding_cache()
        if emb_cache.available:
            cached = emb_cache.get(query)
            if cached:
                logger.debug("Embedding cache HIT | q=%.40s", query)
                return cached

        vec = self.embedding.embed_query(query)

        if emb_cache.available:
            emb_cache.set(query, vec)
        return vec

    def _expand_queries_llm(self, llm, query: str) -> list[str]:
        """LLM Multi-Query 扩展。"""
        try:
            from app.prompts.system_prompts import MULTI_QUERY_PROMPT
            resp = llm.chat([{"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)}])
            variants = [q.strip() for q in resp.strip().split("\n") if q.strip() and q.strip() != query]
            return list(dict.fromkeys(variants))[:3]  # 去重保序
        except Exception:
            return []

    def _plan_decompose(self, llm, query: str) -> list[str]:
        """LLM 拆解复杂问题为子问题。Agent 注入的 planner 优先。"""
        if self._planner:
            try:
                return self._planner.plan(query)
            except Exception:
                logger.exception("Agent planner failed, using default LLM decomposition")

        prompt = (
            "你是一个任务拆解助手。将以下复杂问题拆解为 2-5 个独立的子问题，"
            "每个子问题可以被独立检索和回答。每行一个子问题，不要编号。\n"
            f"复杂问题: {query}\n"
            "子问题:"
        )
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            sub_qs = [q.strip() for q in resp.strip().split("\n") if q.strip()]
            return sub_qs[:5]
        except Exception:
            logger.exception("Plan decomposition LLM failed")
            return []

    # ── Concentration-RRF Fusion ──

    @staticmethod
    def _compute_concentration(raw_results, chunk_map: dict[str, dict]) -> float:
        """计算检索结果的文档集中度。

        一个源如果能"锁定"少数文档，说明检索有效；分散在多个文档说明在碰瓷。
        concentration = 最频繁文档命中的 chunk 数 / 总命中 chunk 数
        范围 [0, 1]，与分数量纲无关。"""
        if not raw_results:
            return 0.0
        doc_counts: dict[str, int] = {}
        total = 0
        for item in raw_results:
            cuid = item[0] if isinstance(item, tuple) else item.get("chunk_uid", "")
            if cuid in chunk_map:
                doc_uid = chunk_map[cuid].get("document_uid", "")
                doc_counts[doc_uid] = doc_counts.get(doc_uid, 0) + 1
                total += 1
        if total == 0:
            return 0.0
        return max(doc_counts.values()) / total

    # 绝对质量门控：防止检索源在没有相关文档时锁定噪声文档
    # Issue 6: 降低阈值以适应跨领域检索（文档是A领域的，用户问的是B领域）
    FAISS_COSINE_MIN = 0.35   # FAISS top-1 cosine 低于此值视为不可信
    ES_BM25_MIN = 2.5         # ES top-1 BM25 低于此值视为不可信

    def _concentration_rrf_fusion(
        self,
        faiss_raw: list[tuple[str, float]],
        es_raw: list[dict],
        chunk_map: dict[str, dict],
    ) -> tuple[list[dict], dict]:
        """浓度加权 RRF 融合 + 绝对质量门控。

        质量门控：FAISS top-1 cosine > 0.35 且 ES top-1 BM25 > 3.0 才参与融合。
        防止"无相关文档时 FAISS 锁定最大文档"的过拟合。

        文档集中度 = 结构性的检索质量信号，与分数量纲无关：
          - 同源高度集中 → 该源找到目标 → 自动获得高 RRF 权重
          - 同源分散命中 → 该源在碰瓷   → 自动获得低 RRF 权重

        返回 (ranked_results, fusion_info)"""
        # 绝对质量检查
        faiss_ok = bool(faiss_raw) and faiss_raw[0][1] > self.FAISS_COSINE_MIN
        es_ok = bool(es_raw) and es_raw[0].get("_score", 0) > self.ES_BM25_MIN

        if not faiss_ok and not es_ok:
            logger.warning(
                "Concentration-RRF: both sources below quality threshold — returning empty, "
                "faiss_top1=%.4f es_top1=%.2f",
                faiss_raw[0][1] if faiss_raw else 0,
                es_raw[0].get("_score", 0) if es_raw else 0,
            )
            return [], {"faiss_ok": False, "es_ok": False, "strategy": "empty"}

        faiss_conc = self._compute_concentration(faiss_raw, chunk_map)
        es_conc = self._compute_concentration(es_raw, chunk_map)

        # 单源可用：直接用该源结果，不融合
        if faiss_ok and not es_ok:
            results = self._single_source_rank(faiss_raw, chunk_map, "faiss")
            logger.info("Concentration-RRF: FAISS-only (es_ok=False), faiss_top1=%.4f faiss_conc=%.2f top5=%s",
                       faiss_raw[0][1], faiss_conc,
                       [r.get("filename", "")[:25] for r in results[:5]])
            return results[:TOP_K], {"faiss_ok": True, "es_ok": False, "faiss_conc": round(faiss_conc, 3),
                                     "strategy": "faiss_only"}

        if es_ok and not faiss_ok:
            results = self._single_source_rank(es_raw, chunk_map, "es")
            logger.info("Concentration-RRF: ES-only (faiss_ok=False), es_top1=%.2f es_conc=%.2f top5=%s",
                       es_raw[0].get("_score", 0), es_conc,
                       [r.get("filename", "")[:25] for r in results[:5]])
            return results[:TOP_K], {"faiss_ok": False, "es_ok": True, "es_conc": round(es_conc, 3),
                                     "strategy": "es_only"}

        # 两个源都通过质量门控 → concentration-RRF
        # 两个源都分散（都不可信）→ 返回空
        if faiss_conc < 0.3 and es_conc < 0.3:
            logger.warning(
                "Concentration-RRF: both sources scattered — returning empty, "
                "faiss_conc=%.2f es_conc=%.2f", faiss_conc, es_conc,
            )
            return [], {"faiss_ok": True, "es_ok": True,
                        "faiss_conc": round(faiss_conc, 3), "es_conc": round(es_conc, 3),
                        "strategy": "empty"}

        # 标准 RRF + concentration 权重
        merged: dict[str, dict] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, (cuid, score) in enumerate(faiss_raw):
            if cuid in chunk_map:
                merged[cuid] = {**chunk_map[cuid], "faiss_score": score, "faiss_rank": rank,
                                "es_score": 0, "es_rank": -1}
                rrf_scores[cuid] += faiss_conc / (RRF_K + rank + 1)

        for rank, hit in enumerate(es_raw):
            cuid = hit.get("chunk_uid", "")
            if cuid not in chunk_map:
                continue
            if cuid not in merged:
                merged[cuid] = {**chunk_map[cuid], "faiss_score": 0, "faiss_rank": -1}
            merged[cuid]["es_score"] = hit.get("_score", 0)
            merged[cuid]["es_rank"] = rank
            rrf_scores[cuid] += es_conc / (RRF_K + rank + 1)

        for cuid, m in merged.items():
            m["rrf_score"] = rrf_scores.get(cuid, 0)

        ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

        fusion_info = {
            "faiss_ok": True, "es_ok": True,
            "faiss_conc": round(faiss_conc, 3),
            "es_conc": round(es_conc, 3),
            "strategy": "concentration_rrf",
        }

        top_fnames = [r.get("filename", "")[:25] for r in ranked[:5]]
        logger.info(
            "Concentration-RRF | faiss_top1=%.4f es_top1=%.2f faiss_conc=%.2f es_conc=%.2f top5=%s",
            faiss_raw[0][1], es_raw[0].get("_score", 0), faiss_conc, es_conc, top_fnames,
        )

        return ranked[:TOP_K], fusion_info

    @staticmethod
    def _single_source_rank(raw_results, chunk_map: dict[str, dict], source: str) -> list[dict]:
        """单源排序：按原始分数降序排列，无融合。"""
        results = []
        for item in raw_results:
            if source == "faiss":
                cuid, score = item
            else:
                cuid = item.get("chunk_uid", "")
                score = item.get("_score", 0)
            if cuid in chunk_map:
                entry = {**chunk_map[cuid]}
                entry["faiss_score"] = score if source == "faiss" else 0
                entry["es_score"] = score if source == "es" else 0
                entry["rrf_score"] = score
                results.append(entry)
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results

    # ── DocSummary 第二层检索 ──

    def _boost_by_doc_summary(self, query: str, results: list[dict], db) -> list[dict]:
        """DocSummaryIndex 第二层检索：文档级匹配提权。

        对抽象/总结类查询，相关文档的 chunks 获得 bonus 分。
        """
        try:
            from app.services.doc_summary_index import get_doc_summary_index

            doc_idx = get_doc_summary_index()
            if doc_idx is None:
                return results

            doc_matches = doc_idx.search(query, k=3)
            if not doc_matches:
                return results

            bonus_doc_uids = {dm[0] for dm in doc_matches}
            boosted = []
            for r in results:
                r = dict(r)
                if r.get("document_uid") in bonus_doc_uids:
                    r["rrf_score"] = r.get("rrf_score", 0) + 0.15  # doc-level bonus
                    r["doc_summary_boost"] = True
                boosted.append(r)

            boosted.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
            logger.debug("DocSummary boost: %d docs matched, %d results re-ranked",
                         len(bonus_doc_uids), len(boosted))
            return boosted
        except Exception:
            logger.debug("DocSummaryIndex unavailable, skipping boost")
            return results

    def _load_chunks(self, db, chunk_uids: list[str], owner_id: int | None = None) -> dict[str, dict]:
        from app.models.document_chunk import DocumentChunk
        from app.models.document import Document

        if not chunk_uids:
            return {}
        q = db.query(DocumentChunk).filter(DocumentChunk.chunk_uid.in_(chunk_uids))
        if owner_id is not None:
            q = q.filter(DocumentChunk.owner_id == owner_id)
        chunks = q.all()
        doc_uids = list(set(c.document_uid for c in chunks))
        docs = {}
        if doc_uids:
            doc_q = db.query(Document).filter(Document.document_uid.in_(doc_uids))
            if owner_id is not None:
                doc_q = doc_q.filter(Document.owner_id == owner_id)
            docs = {
                d.document_uid: d.filename
                for d in doc_q.all()
            }
        result = {}
        for c in chunks:
            result[c.chunk_uid] = {
                "chunk_uid": c.chunk_uid,
                "document_uid": c.document_uid,
                "filename": docs.get(c.document_uid, ""),
                "content": c.content,
                "section_title": c.section_title or "",
                "heading_path": c.heading_path or "",
                "page_no": c.page_no or 0,
                "block_type": c.block_type or "",
                "chunk_index": c.chunk_index,
            }
        return result

    def _enrich_results(self, db, raw_results: list[dict], owner_id: int | None = None) -> list[dict]:
        uids = [h.get("chunk_uid", "") for h in raw_results]
        chunk_map = self._load_chunks(db, uids, owner_id=owner_id)
        results = []
        for hit in raw_results:
            cuid = hit.get("chunk_uid", "")
            if cuid in chunk_map:
                results.append({**chunk_map[cuid], "es_score": hit.get("_score", 0)})
        return results[:TOP_K]

    def _load_cached_results(self, db, chunk_uids: list[str], owner_id: int | None = None) -> list[dict]:
        chunk_map = self._load_chunks(db, chunk_uids, owner_id=owner_id)
        results = []
        for cuid in chunk_uids:
            if cuid in chunk_map:
                results.append(chunk_map[cuid])
        return results[:TOP_K]


def _log_timing_summary(ctx: SearchContext):
    """记录单次检索的耗时拆解。"""
    parts = []
    for k in ("cache", "classify", "plan_llm", "faiss", "es", "rrf", "plan_execute",
              "sub_queries", "reranker", "doc_summary", "total"):
        t = ctx.timings.get(k)
        if t is not None:
            parts.append(f"{k}={t:.0f}ms")
    if ctx.cache_hit:
        parts.append("cache_hit=1")
    logger.info("Timings | %s", " ".join(parts))


# 全局单例
_router: SearchRouter | None = None


def get_search_router() -> SearchRouter:
    global _router
    if _router is None:
        _router = SearchRouter()
    return _router
