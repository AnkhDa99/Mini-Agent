"""
混合搜索：FAISS 向量检索 + Elasticsearch BM25 关键词检索，RRF 融合排序。
"""
import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.es_service import ESService
from app.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)

# RRF 参数
RRF_K = 60
DEFAULT_TOP_K = 10
RETRIEVAL_K = 25  # 每种检索方式取多少条


class HybridSearchService:
    """混合搜索编排器。"""

    def __init__(self):
        self._embedding: EmbeddingService | None = None
        self._vector_store: VectorStore | None = None
        self._es: ESService | None = None
        self._reranker: RerankerService | None = None

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
                logger.info("HybridSearch: ES connected")
            except Exception as e:
                logger.warning("HybridSearch: ES not available: %s", e)
        return self._es

    @property
    def reranker(self) -> RerankerService:
        if self._reranker is None:
            self._reranker = RerankerService()
        return self._reranker

    def search(
        self,
        db: Session,
        query: str,
        project_uid: str = "",
        top_k: int = DEFAULT_TOP_K,
        hyde_query: str = "",
        skip_rerank: bool = False,
        owner_id: int | None = None,
    ) -> list[dict]:
        """
        混合搜索主入口。
        hyde_query: HyDE 生成的假想文档片段，用于 FAISS 向量检索（提升召回率）。
        skip_rerank: Multi-Query 中间轮次跳过精排，由上层统一 rerank。
        """
        # 1. 向量检索：HyDE query 优先，语义更接近真实文档
        vector_query = hyde_query or query
        query_vec = self.embedding.embed_query(vector_query)
        faiss_raw = self.vector_store.search(query_vec, k=RETRIEVAL_K)

        # 2. 关键词检索：始终用原始 query（保留精确关键词匹配能力）
        es_raw: list[dict] = []
        es_svc = self.es
        if es_svc is not None:
            es_raw = es_svc.search(query, project_uid=project_uid, k=RETRIEVAL_K)

        all_chunk_uids: set[str] = set()
        for cuid, _ in faiss_raw:
            all_chunk_uids.add(cuid)
        for hit in es_raw:
            all_chunk_uids.add(hit.get("chunk_uid", ""))

        # 3. 从 MySQL 批量加载 chunk 内容 + 文档元数据
        chunk_map = self._load_chunks(db, list(all_chunk_uids), owner_id=owner_id)

        # 4. 构建结构化结果
        faiss_results: list[dict] = []
        for rank, (cuid, score) in enumerate(faiss_raw):
            if cuid in chunk_map:
                faiss_results.append({**chunk_map[cuid], "faiss_score": score, "faiss_rank": rank})

        es_results: list[dict] = []
        for rank, hit in enumerate(es_raw):
            cuid = hit.get("chunk_uid", "")
            if cuid in chunk_map:
                es_results.append({**chunk_map[cuid], "es_score": hit.get("_score", 0), "es_rank": rank})

        # 5. RRF 融合
        rrf_scores: dict[str, float] = defaultdict(float)
        for r in faiss_results:
            rrf_scores[r["chunk_uid"]] += 1.0 / (RRF_K + r["faiss_rank"] + 1)
        for r in es_results:
            rrf_scores[r["chunk_uid"]] += 1.0 / (RRF_K + r["es_rank"] + 1)

        # 合并去重
        merged: dict[str, dict] = {}
        for r in faiss_results:
            merged[r["chunk_uid"]] = r
        for r in es_results:
            cuid = r["chunk_uid"]
            if cuid not in merged:
                merged[cuid] = r
            else:
                merged[cuid]["es_score"] = r.get("es_score", 0)
                merged[cuid]["es_rank"] = r.get("es_rank", -1)

        for cuid, m in merged.items():
            m["rrf_score"] = rrf_scores.get(cuid, 0)
            m["faiss_score"] = m.get("faiss_score", 0)
            m["es_score"] = m.get("es_score", 0)

        # 按 RRF 分数排序
        ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

        # ---- 在线诊断：检索质量信号 ----
        top1_rrf = round(ranked[0]["rrf_score"], 5) if ranked else 0
        top1_faiss = round(faiss_raw[0][1], 4) if faiss_raw else 0
        top1_es = round(es_raw[0].get("_score", 0), 2) if es_raw else 0

        top3_filenames = [r.get("filename", "")[:30] for r in ranked[:3]]

        # chunk/文档级 overlap（仅在 ES 有结果时才有意义）
        faiss_docs = {chunk_map[cuid]["document_uid"] for cuid, _ in faiss_raw if cuid in chunk_map}
        es_docs: set[str] = set()
        es_contributing = bool(es_raw)
        if es_contributing:
            es_docs = {chunk_map[cuid]["document_uid"] for cuid in {hit.get("chunk_uid", "") for hit in es_raw} if cuid in chunk_map}
            faiss_uids = {cuid for cuid, _ in faiss_raw}
            es_uids = {hit.get("chunk_uid", "") for hit in es_raw}
            chunk_overlap = round(len(faiss_uids & es_uids) / len(faiss_uids | es_uids) * 100, 1) if (faiss_uids | es_uids) else 0
            doc_overlap = round(len(faiss_docs & es_docs) / len(faiss_docs | es_docs) * 100, 1) if (faiss_docs | es_docs) else 0
            doc_intersection = faiss_docs & es_docs
            logger.info(
                "Search quality | query=%.40s | "
                "chunk_overlap=%.1f%% doc_overlap=%.1f%% "
                "(faiss_docs=%d es_docs=%d shared_docs=%d) | "
                "rrf_top1=%.5f faiss_top1=%.4f es_top1=%.2f hyde=%s | "
                "top3=%s",
                query, chunk_overlap, doc_overlap,
                len(faiss_docs), len(es_docs), len(doc_intersection),
                top1_rrf, top1_faiss, top1_es, bool(hyde_query),
                top3_filenames,
            )
        else:
            logger.info(
                "Search quality | query=%.40s | "
                "faiss_docs=%d es=UNAVAILABLE | "
                "rrf_top1=%.5f faiss_top1=%.4f hyde=%s | "
                "top3=%s",
                query, len(faiss_docs),
                top1_rrf, top1_faiss, bool(hyde_query),
                top3_filenames,
            )

        # 质量告警
        if top1_faiss < 0.3 and faiss_raw:
            logger.warning(
                "Search quality WARNING | query=%.40s | FAISS_WEAK: top1 %.4f < 0.3 | rrf=%.5f es=%.2f",
                query, top1_faiss, top1_rrf, top1_es,
            )

        # ---- 精排：Cross-encoder reranker（Multi-Query 中间轮次跳过） ----
        if not skip_rerank:
            ranked = self.reranker.rerank(query, ranked, top_k=top_k)

        return ranked[:top_k]

    def _load_chunks(self, db: Session, chunk_uids: list[str], owner_id: int | None = None) -> dict[str, dict]:
        """批量加载 chunk 内容和关联文档文件名。"""
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
                "project_uid": c.project_uid,
                "filename": docs.get(c.document_uid, ""),
                "content": c.content,
                "section_title": c.section_title or "",
                "heading_path": c.heading_path or "",
                "page_no": c.page_no or 0,
                "block_type": c.block_type or "",
                "chunk_index": c.chunk_index,
            }
        return result
