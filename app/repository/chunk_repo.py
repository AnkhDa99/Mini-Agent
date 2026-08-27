from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def bulk_create_chunks(db: Session, chunks: list[DocumentChunk]):
    if not chunks:
        return []

    db.add_all(chunks)
    db.commit()

    for chunk in chunks:
        db.refresh(chunk)

    return chunks


def count_chunks_by_document(db: Session, document_uid: str, version: int = 1) -> int:
    return (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_uid == document_uid,
            DocumentChunk.version == version,
        )
        .count()
    )

def delete_chunks_by_document_uid(db, document_uid: str) -> int:
    count = db.query(DocumentChunk).filter(
        DocumentChunk.document_uid == document_uid
    ).delete(synchronize_session=False)

    db.commit()
    return count

def list_chunks_by_document(
    db: Session,
    document_uid: str,
    version: int = 1,
    limit: int = 20,
):
    return (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_uid == document_uid,
            DocumentChunk.version == version,
        )
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(limit)
        .all()
    )


def list_chunks_pending_embedding(db: Session, limit: int = 100):
    """获取 embedding_status=pending 的分块，用于批量向量化。"""
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.embedding_status == "pending")
        .order_by(DocumentChunk.id.asc())
        .limit(limit)
        .all()
    )


def update_chunk_embedding_done(db: Session, chunk_uids: list[str]):
    """批量更新分块 embedding_status → done。"""
    if not chunk_uids:
        return
    (
        db.query(DocumentChunk)
        .filter(DocumentChunk.chunk_uid.in_(chunk_uids))
        .update(
            {"embedding_status": "done"},
            synchronize_session=False,
        )
    )
    db.commit()


def list_chunks_by_uids(db: Session, chunk_uids: list[str]):
    """根据 chunk_uid 列表批量查询分块。"""
    if not chunk_uids:
        return []
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.chunk_uid.in_(chunk_uids))
        .all()
    )