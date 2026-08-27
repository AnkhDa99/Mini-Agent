from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_parse_task import DocumentParseTask


def get_document_by_uid(db: Session, document_uid: str):
    return (
        db.query(Document)
        .filter(Document.document_uid == document_uid)
        .first()
    )


def create_document(
    db: Session,
    document_uid: str,
    project_uid: str,
    filename: str,
    file_md5: str,
    file_size: int,
    file_type: str | None,
    bucket: str,
    object_key: str,
    owner_id: int | None = None,
):
    obj = Document(
        document_uid=document_uid,
        project_uid=project_uid,
        owner_id=owner_id,
        filename=filename,
        file_md5=file_md5,
        file_size=file_size,
        file_type=file_type,
        bucket=bucket,
        object_key=object_key,
        upload_status="success",
        parse_status="pending",
        chunk_status="pending",
        embedding_status="pending",
        index_status="pending",
        available_for_search=False,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def create_document_parse_task(
    db: Session,
    task_uid: str,
    document_uid: str,
    max_retry: int = 5,
):
    obj = DocumentParseTask(
        task_uid=task_uid,
        document_uid=document_uid,
        status="pending",
        retry_count=0,
        max_retry=max_retry,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_parse_task_by_document_uid(db: Session, document_uid: str):
    return (
        db.query(DocumentParseTask)
        .filter(DocumentParseTask.document_uid == document_uid)
        .order_by(DocumentParseTask.id.desc())
        .first()
    )

def update_document_parse_success(db: Session, document_uid: str):
    doc = get_document_by_uid(db, document_uid)
    if doc:
        doc.parse_status = "success"
        doc.chunk_status = "success"
        doc.last_error = None
        db.commit()
        db.refresh(doc)
    return doc


def update_document_parse_failed(db: Session, document_uid: str, error: str):
    doc = get_document_by_uid(db, document_uid)
    if doc:
        doc.parse_status = "failed"
        doc.chunk_status = "failed"
        doc.last_error = error
        db.commit()
        db.refresh(doc)
    return doc

def reset_document_parse_status(db, document_uid: str):
    doc = get_document_by_uid(db, document_uid)
    if not doc:
        return None

    doc.parse_status = "pending"
    doc.chunk_status = "pending"
    doc.embedding_status = "pending"
    doc.index_status = "pending"
    doc.available_for_search = False
    doc.last_error = None

    db.commit()
    db.refresh(doc)
    return doc


def update_document_embedding_success(db: Session, document_uid: str):
    doc = get_document_by_uid(db, document_uid)
    if doc:
        doc.embedding_status = "success"
        doc.index_status = "success"
        doc.available_for_search = True
        doc.last_error = None
        db.commit()
        db.refresh(doc)
    return doc


def update_document_embedding_failed(db: Session, document_uid: str, error: str):
    doc = get_document_by_uid(db, document_uid)
    if doc:
        doc.embedding_status = "failed"
        doc.last_error = error
        db.commit()
        db.refresh(doc)
    return doc


def list_documents_by_owner(db: Session, owner_id: int, limit: int = 50):
    """列出用户拥有的文档。"""
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .all()
    )


def delete_document(db: Session, document_uid: str) -> bool:
    """删除文档记录，返回是否成功。"""
    doc = get_document_by_uid(db, document_uid)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True


def list_documents_pending_index(db: Session, limit: int = 50):
    """获取 chunk_status=success 但 embedding_status=pending 的文档。"""
    return (
        db.query(Document)
        .filter(
            Document.chunk_status == "success",
            Document.embedding_status == "pending",
        )
        .order_by(Document.id.asc())
        .limit(limit)
        .all()
    )