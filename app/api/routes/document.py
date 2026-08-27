from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.services.upload_service import UploadService

from fastapi import File, Form, UploadFile
from app.schemas.document import (
    InitUploadRequest,
    InitUploadResponse,
    UploadPartResponse,
    UploadStatusResponse,
)

from app.schemas.document import (
    InitUploadRequest,
    InitUploadResponse,
    UploadPartResponse,
    UploadStatusResponse,
    CompleteUploadRequest,
    CompleteUploadResponse,
    DocumentStatusResponse,
)

from app.schemas.document import CompleteUploadRequest, CompleteUploadResponse

router = APIRouter(tags=["documents"])
service = UploadService()


@router.get("/documents")
def list_my_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户的文档（管理员可看到全部含历史 NULL owner 数据）。"""
    from app.models.document import Document
    if current_user.role == "admin":
        docs = (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .all()
        )
    else:
        docs = (
            db.query(Document)
            .filter(Document.owner_id == current_user.id)
            .order_by(Document.created_at.desc())
            .all()
        )
    return [
        {
            "document_uid": d.document_uid,
            "filename": d.filename,
            "file_size": d.file_size,
            "file_type": d.file_type,
            "parse_status": d.parse_status,
            "chunk_status": d.chunk_status,
            "embedding_status": d.embedding_status,
            "available_for_search": d.available_for_search,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.post("/documents/init_upload", response_model=InitUploadResponse)
def init_upload(
    req: InitUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 游客文档数量限制
    if current_user.role == "guest":
        from app.core.config import settings
        from app.repository.document_repo import list_documents_by_owner
        existing_docs = list_documents_by_owner(db, current_user.id)
        if len(existing_docs) >= (current_user.document_limit or settings.guest_document_limit):
            raise HTTPException(
                status_code=403,
                detail=f"您已上传{len(existing_docs)}个文档，达到游客上限。请联系管理员获取正式账号：{settings.contact_admin_email}"
            )
    try:
        return service.init_upload(db=db, req=req, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"初始化上传失败：{exc}") from exc

@router.post("/documents/upload_part", response_model=UploadPartResponse)
async def upload_part(
    upload_uid: str = Form(...),
    part_number: int = Form(...),
    part_md5: str = Form(...),
    file_part: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return await service.upload_part(
            db=db,
            upload_uid=upload_uid,
            part_number=part_number,
            part_md5=part_md5,
            file_part=file_part,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传分片失败：{exc}") from exc

@router.get("/documents/{upload_uid}/upload_status", response_model=UploadStatusResponse)
def upload_status(
    upload_uid: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return service.get_upload_status(db=db, upload_uid=upload_uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询上传状态失败：{exc}") from exc

@router.post("/documents/complete_upload", response_model=CompleteUploadResponse)
def complete_upload(
    req: CompleteUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.complete_upload(db=db, req=req, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"完成上传失败：{exc}") from exc

@router.get("/documents/{document_uid}/status", response_model=DocumentStatusResponse)
def document_status(
    document_uid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.get_document_status(db=db, document_uid=document_uid, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询文档状态失败：{exc}") from exc


@router.delete("/documents/{document_uid}")
def delete_document(
    document_uid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """级联删除文档：MySQL chunks → ES → FAISS → MinIO → MySQL document。"""
    try:
        result = service.delete_document(db=db, document_uid=document_uid, user=current_user)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败：{exc}") from exc


@router.post("/documents/{upload_uid}/cancel")
def cancel_upload(
    upload_uid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """取消上传会话，清理已上传的分片、文档、chunks、向量。"""
    try:
        return service.cancel_upload(db=db, upload_uid=upload_uid, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"取消失败：{exc}") from exc


@router.post("/admin/cleanup-expired-uploads")
def trigger_cleanup_expired_uploads(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """手动触发清理过期的上传会话及 MinIO 临时分片/孤儿文件。"""
    try:
        cleaned = service.cleanup_expired_sessions(db)
        return {"cleaned": cleaned}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"清理失败：{exc}") from exc