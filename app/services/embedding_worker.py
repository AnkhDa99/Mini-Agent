import logging
import time as _time

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repository.chunk_repo import (
    list_chunks_pending_embedding,
    update_chunk_embedding_done,
    count_chunks_by_document,
)
from app.repository.document_repo import (
    update_document_embedding_success,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.es_service import ESService

logger = logging.getLogger(__name__)

_scheduler = None

_embedding_service = None
_vector_store = None
_es_service = None
_es_last_attempt = 0
_ES_RETRY_INTERVAL = 30  # ES 重连间隔（秒）


def _get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_es_service():
    """获取 ES 服务实例，失败时每 30 秒重试一次。首次创建时若索引重建，自动触发全量回填。"""
    global _es_service, _es_last_attempt
    now = _time.time()
    if _es_service is None and (now - _es_last_attempt) > _ES_RETRY_INTERVAL:
        _es_last_attempt = now
        try:
            _es_service = ESService()
            logger.info("ES 连接成功，BM25 索引已启用")
            if _es_service.needs_reindex:
                _trigger_full_reindex()
        except Exception as e:
            logger.warning("ES 未就绪: %s，将在 %d 秒后重试", e, _ES_RETRY_INTERVAL)
    return _es_service


def _trigger_full_reindex():
    """索引重建后，将 MySQL 中所有 chunk 的 index_status 重置为 pending，由 backfill 回填 ES。"""
    db = SessionLocal()
    try:
        updated = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.index_status == "done")
            .update({"index_status": "pending"}, synchronize_session=False)
        )
        db.commit()
        logger.info("ES index rebuilt: reset %d chunks to pending for backfill", updated)
    except Exception:
        logger.exception("Failed to reset index_status for ES rebuild")
    finally:
        db.close()


def _chunk_to_es_doc(chunk, filename: str = "") -> dict:
    return {
        "chunk_uid": chunk.chunk_uid,
        "document_uid": chunk.document_uid,
        "project_uid": chunk.project_uid,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "section_title": chunk.section_title or "",
        "heading_path": chunk.heading_path or "",
        "block_type": chunk.block_type or "",
        "page_no": chunk.page_no or 0,
        "filename": filename,
    }


def _update_chunk_index_done(db, chunk_uids: list[str]):
    """批量更新 index_status → done。"""
    if not chunk_uids:
        return
    (
        db.query(DocumentChunk)
        .filter(DocumentChunk.chunk_uid.in_(chunk_uids))
        .update({"index_status": "done"}, synchronize_session=False)
    )
    db.commit()


def _load_filenames(db, chunks: list) -> dict[str, str]:
    """从 Document 表加载文件名映射。"""
    doc_uids = list(set(c.document_uid for c in chunks))
    if not doc_uids:
        return {}
    docs = (
        db.query(Document.document_uid, Document.filename)
        .filter(Document.document_uid.in_(doc_uids))
        .all()
    )
    return {d.document_uid: d.filename for d in docs}


def _backfill_es_index(db, limit: int = 50) -> int:
    """
    兜底：将 embedding_status=done 但 index_status!=done 的 chunk 补索引到 ES。
    处理 ES 首次不可用导致的漏索引问题。
    """
    es_svc = _get_es_service()
    if es_svc is None:
        return 0

    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.embedding_status == "done",
            DocumentChunk.index_status != "done",
        )
        .order_by(DocumentChunk.id.asc())
        .limit(limit)
        .all()
    )
    if not chunks:
        return 0

    filenames = _load_filenames(db, chunks)
    es_docs = [_chunk_to_es_doc(c, filenames.get(c.document_uid, "")) for c in chunks]
    indexed = es_svc.index_chunks(es_docs)

    if indexed > 0:
        _update_chunk_index_done(db, [c.chunk_uid for c in chunks[:indexed]])

    return indexed


def process_embedding_batch():
    """
    单次批处理：
    1. 取 pending chunks → 2. 生成 embedding →
    3. 写入 FAISS + ES → 4. 更新 chunk embedding_status + index_status → done →
    5. 检查文档是否全部完成，更新 document 状态
    6. 兜底补索引：已向量化但 ES 未索引的 chunk
    """
    db = SessionLocal()
    try:
        # ---- 主流程：pending → embedding → FAISS + ES ----
        chunks = list_chunks_pending_embedding(db, limit=settings.embedding_worker_batch_size)
        if chunks:
            logger.info("Embedding worker: processing %d chunks", len(chunks))

            texts = [c.content for c in chunks]
            try:
                vectors = _get_embedding_service().embed_texts(texts)
            except Exception:
                logger.exception("Embedding generation failed, skipping batch")
                return

            chunk_uids = [c.chunk_uid for c in chunks]
            _get_vector_store().add(chunk_uids, vectors)

            # ES 索引（best-effort，失败由 backfill 兜底）
            es_svc = _get_es_service()
            if es_svc is not None:
                filenames = _load_filenames(db, chunks)
                es_docs = [_chunk_to_es_doc(c, filenames.get(c.document_uid, "")) for c in chunks]
                indexed = es_svc.index_chunks(es_docs)
                if indexed > 0:
                    _update_chunk_index_done(db, [c.chunk_uid for c in chunks[:indexed]])

            update_chunk_embedding_done(db, chunk_uids)
            _update_document_status(db, chunks)

        # ---- 兜底：补索引漏掉的 ES chunk ----
        backfilled = _backfill_es_index(db, limit=settings.embedding_worker_batch_size)
        if backfilled > 0:
            logger.info("Embedding worker: backfilled %d chunks to ES", backfilled)

    except Exception:
        logger.exception("Embedding worker batch failed")
    finally:
        db.close()


def _update_document_status(db, chunks):
    """检查本次涉及的文档是否所有 chunk 都已完成 embedding。"""
    doc_uids = set(c.document_uid for c in chunks)
    for doc_uid in doc_uids:
        total = count_chunks_by_document(db, doc_uid)
        if total == 0:
            continue
        done_count = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_uid == doc_uid,
                DocumentChunk.embedding_status == "done",
            )
            .count()
        )
        if done_count >= total:
            try:
                update_document_embedding_success(db, doc_uid)
                logger.info("Document embedding complete: %s", doc_uid)
            except Exception:
                logger.exception("Failed to update document status: %s", doc_uid)


def start_embedding_worker():
    global _scheduler

    if _scheduler is not None:
        logger.info("Embedding worker already running")
        return

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # interval 从 3s → 5s，减少批量重叠导致的 skipped 警告
    scheduler.add_job(
        process_embedding_batch,
        trigger="interval",
        seconds=5,
        id="embedding_worker_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    _scheduler = scheduler

    logger.info("Embedding worker started | interval=5s fetch_batch=%d api_batch=%d",
                settings.embedding_worker_batch_size, settings.embedding_batch_size)


def stop_embedding_worker():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Embedding worker stopped")
