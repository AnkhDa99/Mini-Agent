from sqlalchemy.orm import Session

from app.models.upload_session import UploadSession

from app.models.upload_part import UploadPart
from app.models.upload_session import UploadSession

from app.models.upload_session import (
    UPLOAD_STATUS_UPLOADING,
    UPLOAD_STATUS_MERGING,
    UPLOAD_STATUS_COMPLETED,
    UPLOAD_STATUS_FAILED,
)

def get_upload_session_by_uid(db: Session, upload_uid: str):
    return (
        db.query(UploadSession)
        .filter(UploadSession.upload_uid == upload_uid)
        .first()
    )


def get_latest_upload_session_by_file(
    db: Session,
    project_uid: str,
    file_md5: str,
    filename: str,
):
    return (
        db.query(UploadSession)
        .filter(
            UploadSession.project_uid == project_uid,
            UploadSession.file_md5 == file_md5,
            UploadSession.filename == filename,
        )
        .order_by(UploadSession.id.desc())
        .first()
    )


def create_upload_session(
    db: Session,
    upload_uid: str,
    project_uid: str,
    filename: str,
    file_md5: str,
    file_size: int,
    content_type: str | None,
    chunk_size: int,
    total_parts: int,
    expires_at,
    owner_id: int | None = None,
):
    obj = UploadSession(
        upload_uid=upload_uid,
        project_uid=project_uid,
        owner_id=owner_id,
        filename=filename,
        file_md5=file_md5,
        file_size=file_size,
        content_type=content_type,
        chunk_size=chunk_size,
        total_parts=total_parts,
        uploaded_parts=0,
        status="uploading",
        expires_at=expires_at,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_upload_part(db: Session, upload_uid: str, part_number: int):
    return (
        db.query(UploadPart)
        .filter(
            UploadPart.upload_uid == upload_uid,
            UploadPart.part_number == part_number,
        )
        .first()
    )


def create_or_update_upload_part(
    db: Session,
    upload_uid: str,
    part_number: int,
    part_size: int,
    part_md5: str,
    object_key: str,
):
    obj = get_upload_part(db, upload_uid, part_number)

    if obj is None:
        obj = UploadPart(
            upload_uid=upload_uid,
            part_number=part_number,
            part_size=part_size,
            part_md5=part_md5,
            object_key=object_key,
            status="success",
        )
        db.add(obj)
    else:
        obj.part_size = part_size
        obj.part_md5 = part_md5
        obj.object_key = object_key
        obj.status = "success"

    db.commit()
    db.refresh(obj)
    return obj


def list_success_part_numbers(db: Session, upload_uid: str) -> list[int]:
    rows = (
        db.query(UploadPart.part_number)
        .filter(
            UploadPart.upload_uid == upload_uid,
            UploadPart.status == "success",
        )
        .order_by(UploadPart.part_number.asc())
        .all()
    )
    return [row[0] for row in rows]


def update_uploaded_parts_count(db: Session, upload_uid: str, uploaded_parts: int):
    obj = get_upload_session_by_uid(db, upload_uid)
    if obj:
        obj.uploaded_parts = uploaded_parts
        db.commit()
        db.refresh(obj)
    return obj

def list_success_parts(db: Session, upload_uid: str):
    return (
        db.query(UploadPart)
        .filter(
            UploadPart.upload_uid == upload_uid,
            UploadPart.status == "success",
        )
        .order_by(UploadPart.part_number.asc())
        .all()
    )

def mark_upload_merging(db: Session, upload_uid: str) -> bool:
    """
    CAS 抢占合并权：只有 uploading 状态可以变成 merging。
    返回 True 表示当前请求抢占成功。
    """
    affected = (
        db.query(UploadSession)
        .filter(
            UploadSession.upload_uid == upload_uid,
            UploadSession.status == UPLOAD_STATUS_UPLOADING,
        )
        .update(
            {
                UploadSession.status: UPLOAD_STATUS_MERGING,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return affected == 1


def mark_upload_completed(db: Session, upload_uid: str, document_uid: str):
    obj = get_upload_session_by_uid(db, upload_uid)
    if obj:
        obj.status = UPLOAD_STATUS_COMPLETED
        obj.document_uid = document_uid
        obj.last_error = None
        db.commit()
        db.refresh(obj)
    return obj


def mark_upload_failed(db: Session, upload_uid: str, error: str):
    obj = get_upload_session_by_uid(db, upload_uid)
    if obj:
        obj.status = UPLOAD_STATUS_FAILED
        obj.last_error = error
        db.commit()
        db.refresh(obj)
    return obj


def get_expired_upload_sessions(db: Session):
    """查询已过期且未完成/未标记过期的上传会话。"""
    from app.core.time_utils import now_shanghai
    from app.models.upload_session import (
        UPLOAD_STATUS_UPLOADING,
        UPLOAD_STATUS_MERGING,
        UPLOAD_STATUS_FAILED,
    )

    return (
        db.query(UploadSession)
        .filter(
            UploadSession.expires_at < now_shanghai(),
            UploadSession.status.in_([
                UPLOAD_STATUS_UPLOADING,
                UPLOAD_STATUS_MERGING,
                UPLOAD_STATUS_FAILED,
            ]),
        )
        .all()
    )


def mark_upload_expired(db: Session, upload_uid: str):
    from app.models.upload_session import UPLOAD_STATUS_EXPIRED

    obj = get_upload_session_by_uid(db, upload_uid)
    if obj:
        obj.status = UPLOAD_STATUS_EXPIRED
        db.commit()
        db.refresh(obj)
    return obj