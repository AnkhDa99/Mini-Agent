"""
文档摘要索引 — 第二层检索。

为每个已索引文档生成摘要向量，概念/抽象查询先用摘要索引定位文档，
再在定位到的文档内做 chunk 级精确检索。

双层检索：Query → 摘要索引（定位文档） → chunk 索引（精确检索）

为 Agent 预留接口：
- Agent 可注入自定义摘要生成逻辑
- Agent 可扩展索引构建策略
"""
import logging
from collections import defaultdict

import numpy as np

from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DocSummaryIndex:
    """文档级摘要向量索引。

    轻量实现：内存中维护文档摘要向量 + FAISS 索引。
    数据量小时（< 1000 文档）直接用 numpy 矩阵检索。

    Agent 扩展点:
    - summary_builder(doc_uid, chunks) → str: 自定义摘要生成
    - build_hook(): 索引构建后回调
    """

    def __init__(self):
        self._embedding: EmbeddingService | None = None
        self._doc_vectors: np.ndarray | None = None       # [n_docs, dim]
        self._doc_uids: list[str] = []                     # 对齐 _doc_vectors
        self._doc_summaries: dict[str, str] = {}            # doc_uid → 摘要文本
        self._built = False

        # Agent 扩展
        self.summary_builder = None  # Callable[[str, list[dict]], str]
        self.build_hook = None       # Callable[[], None]

    @property
    def embedding(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    @property
    def is_ready(self) -> bool:
        return self._built and self._doc_vectors is not None and len(self._doc_vectors) > 0

    def build(self, db, force: bool = False):
        """从 MySQL 构建文档摘要索引。"""
        if self._built and not force:
            return

        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk

        docs = db.query(Document).all()
        if not docs:
            logger.warning("No documents to build summary index")
            return

        doc_chunks: dict[str, list] = defaultdict(list)
        chunks = db.query(DocumentChunk).all()
        for c in chunks:
            doc_chunks[c.document_uid].append(c)

        summaries: dict[str, str] = {}

        for doc in docs:
            chunks_list = doc_chunks.get(doc.document_uid, [])
            if not chunks_list:
                continue

            if self.summary_builder:
                summary = self.summary_builder(doc.document_uid, chunks_list)
            else:
                summary = self._build_summary_heuristic(doc, chunks_list)

            if summary:
                summaries[doc.document_uid] = summary

        if not summaries:
            logger.warning("No summaries generated")
            return

        # 向量化
        doc_uids = list(summaries.keys())
        summary_texts = [summaries[uid] for uid in doc_uids]
        vectors = []
        for text in summary_texts:
            vec = self.embedding.embed_query(text)
            vectors.append(vec)

        self._doc_vectors = np.array(vectors, dtype=np.float32)
        self._doc_uids = doc_uids
        self._doc_summaries = summaries
        self._built = True

        # 归一化（cosine similarity）
        norms = np.linalg.norm(self._doc_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._doc_vectors = self._doc_vectors / norms

        logger.info(
            "DocSummaryIndex built | %d documents | dim=%d",
            len(doc_uids), self._doc_vectors.shape[1],
        )

        if self.build_hook:
            self.build_hook()

    def search(self, query: str, k: int = 5) -> list[tuple[str, float, str]]:
        """搜索最相关的文档摘要。返回 [(doc_uid, score, summary), ...]"""
        if not self.is_ready:
            return []

        query_vec = np.array(self.embedding.embed_query(query), dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)

        scores = np.dot(self._doc_vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_uid = self._doc_uids[idx]
                results.append((
                    doc_uid,
                    float(scores[idx]),
                    self._doc_summaries.get(doc_uid, ""),
                ))
        return results

    @staticmethod
    def _build_summary_heuristic(doc, chunks: list) -> str:
        """启发式摘要生成：文件名 + 前 3 个章节标题 + 前 2 段内容摘要。"""
        parts = [doc.filename or ""]

        # 章节标题
        sections = []
        for c in chunks:
            if c.section_title and c.section_title not in sections:
                sections.append(c.section_title)
        if sections:
            parts.append("章节: " + ", ".join(sections[:5]))

        # 前 2 段内容摘要
        paragraphs = [c for c in chunks if c.block_type == "paragraph"]
        for p in paragraphs[:2]:
            content = (p.content or "")[:200]
            if content:
                parts.append(content)

        return " | ".join(parts)


# 全局单例
_doc_summary_index: DocSummaryIndex | None = None


def get_doc_summary_index() -> DocSummaryIndex:
    global _doc_summary_index
    if _doc_summary_index is None:
        _doc_summary_index = DocSummaryIndex()
    return _doc_summary_index
