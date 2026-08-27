import math
import uuid

from sqlalchemy.orm import Session
from app.models.document import Document
from app.repository.chunk_repo import delete_chunks_by_document_uid

from app.repository.document_repo import reset_document_parse_status

from app.core.config import settings
from app.core.redis_client import redis_client
from app.core.time_utils import now_shanghai
from datetime import timedelta

from app.repository.document_repo import get_document_by_uid
from app.schemas.document import DocumentStatusResponse

from app.repository.document_repo import get_parse_task_by_document_uid

from app.schemas.document import (
    InitUploadRequest,
    InitUploadResponse,
    UploadPartResponse,
    UploadStatusResponse,
)
from app.repository.upload_repo import list_success_parts

from app.repository.upload_repo import (
    get_latest_upload_session_by_file,
    create_upload_session,
)

import hashlib
import os
import tempfile
from fastapi import UploadFile

from app.models.upload_session import UPLOAD_STATUS_UPLOADING, UploadSession
from app.repository.upload_repo import (
    get_upload_session_by_uid,
    create_or_update_upload_part,
    list_success_part_numbers,
    update_uploaded_parts_count,
)
from app.schemas.document import UploadPartResponse
from app.storage.minio_client import minio_storage

import shutil

from app.schemas.document import CompleteUploadRequest, CompleteUploadResponse
from app.repository.upload_repo import (
    mark_upload_merging,
    mark_upload_completed,
    mark_upload_failed,
    get_expired_upload_sessions,
    mark_upload_expired,
)
from app.repository.document_repo import (
    create_document,
    create_document_parse_task,
)
from app.mq.kafka_producer import send_document_parse_message


class UploadService:
    def init_upload(self, db: Session, req: InitUploadRequest, user=None) -> InitUploadResponse:
        filename = req.filename.strip()
        if not filename:
            raise ValueError("文件名不能为空")

        ext = os.path.splitext(filename)[1].lower()

        allowed_exts = {
            ".txt", ".md", ".markdown",
            ".pdf",
            ".doc", ".docx",
            ".ppt", ".pptx",
            ".xls", ".xlsx", ".xlsm",
            ".csv",
            ".html", ".htm",
            ".json",
            ".xml",
            ".yaml", ".yml",
            ".log",
            ".py", ".java", ".js", ".ts", ".vue", ".css", ".sql",
        }

        if ext not in allowed_exts:
            raise ValueError(
                f"暂不支持该文件类型：{ext}。"
                f"当前支持 txt/md/pdf/docx/pptx/xlsx/csv/html/json/xml/yaml/log/常见代码文件"
            )

        if req.file_size > settings.upload_max_file_size_bytes:
            raise ValueError(
                f"文件过大，最大允许 {settings.upload_max_file_size_bytes} 字节"
            )

        chunk_size = req.chunk_size or settings.upload_chunk_size_bytes
        if chunk_size <= 0:
            raise ValueError("chunk_size 不合法")

        total_parts = math.ceil(req.file_size / chunk_size)
        if total_parts <= 0:
            raise ValueError("total_parts 计算异常")

        # 幂等处理：同一个项目下，同名同 MD5 文件，复用已有上传会话
        existed = get_latest_upload_session_by_file(
            db=db,
            project_uid=req.project_uid,
            file_md5=req.file_md5,
            filename=filename,
        )

        if existed and existed.status in {"uploading", "merging", "completed"}:
            if existed.status == "completed" and existed.document_uid:
                doc = get_document_by_uid(db, existed.document_uid)
                parse_task = get_parse_task_by_document_uid(db, existed.document_uid)

                # 情况 1：文档存在，并且解析已经成功，直接复用
                if doc and doc.parse_status == "success" and doc.chunk_status == "success":
                    return InitUploadResponse(
                        upload_uid=existed.upload_uid,
                        filename=existed.filename,
                        file_size=existed.file_size,
                        file_md5=existed.file_md5,
                        chunk_size=existed.chunk_size,
                        total_parts=existed.total_parts,
                        uploaded_parts=list(range(existed.total_parts)),
                        missing_parts=[],
                        status=existed.status,
                        document_uid=existed.document_uid,
                        parse_task_uid=parse_task.task_uid if parse_task else None,
                        reused=True,
                        message="文件已上传并解析完成，复用已有解析结果",
                    )

                # 情况 2：文件已上传完成，但文档解析失败或未完成
                # 不再重复上传分片，也不重新合并文件，而是复用 MinIO 原始文件重新投递解析任务
                if doc and (doc.parse_status == "failed" or doc.chunk_status == "failed"):
                    delete_chunks_by_document_uid(db, doc.document_uid)
                    reset_document_parse_status(db, doc.document_uid)
                    new_parse_task_uid = "parse_" + uuid.uuid4().hex[:24]

                    new_parse_task = create_document_parse_task(
                        db=db,
                        task_uid=new_parse_task_uid,
                        document_uid=doc.document_uid,
                    )

                    try:
                        send_document_parse_message(
                            {
                                "task_uid": new_parse_task.task_uid,
                                "document_uid": doc.document_uid,
                                "object_key": doc.object_key,
                            }
                        )
                    except Exception as exc:
                        existed.last_error = f"Kafka parse message resend failed: {exc}"
                        db.commit()

                    return InitUploadResponse(
                        upload_uid=existed.upload_uid,
                        filename=existed.filename,
                        file_size=existed.file_size,
                        file_md5=existed.file_md5,
                        chunk_size=existed.chunk_size,
                        total_parts=existed.total_parts,
                        uploaded_parts=list(range(existed.total_parts)),
                        missing_parts=[],
                        status=existed.status,
                        document_uid=doc.document_uid,
                        parse_task_uid=new_parse_task.task_uid,
                        reused=True,
                        message="文件已上传过，但上次解析失败，已重新提交解析任务",
                    )

                # 情况 3：文件已上传完成，但 document 不存在或状态不明确
                # 先返回已有 document_uid，让前端继续查询；后续可以做补偿任务
                return InitUploadResponse(
                    upload_uid=existed.upload_uid,
                    filename=existed.filename,
                    file_size=existed.file_size,
                    file_md5=existed.file_md5,
                    chunk_size=existed.chunk_size,
                    total_parts=existed.total_parts,
                    uploaded_parts=list(range(existed.total_parts)),
                    missing_parts=[],
                    status=existed.status,
                    document_uid=existed.document_uid,
                    parse_task_uid=parse_task.task_uid if parse_task else None,
                    reused=True,
                    message="文件已上传完成，正在复用已有文档状态",
                )

            # uploading / merging 状态才需要计算缺失分片
            uploaded_parts = self._get_uploaded_parts_from_redis(
                upload_uid=existed.upload_uid,
                total_parts=existed.total_parts,
            )

            # Redis 可能丢失，MySQL 兜底补充分片状态
            db_parts = list_success_part_numbers(db, existed.upload_uid)
            uploaded_parts = sorted(set(uploaded_parts) | set(db_parts))

            missing_parts = [
                i for i in range(existed.total_parts)
                if i not in set(uploaded_parts)
            ]

            return InitUploadResponse(
                upload_uid=existed.upload_uid,
                filename=existed.filename,
                file_size=existed.file_size,
                file_md5=existed.file_md5,
                chunk_size=existed.chunk_size,
                total_parts=existed.total_parts,
                uploaded_parts=uploaded_parts,
                missing_parts=missing_parts,
                status=existed.status,
                document_uid=existed.document_uid,
                reused=True,
                message="该文件已上传，复用该文件",
            )

        upload_uid = "upload_" + uuid.uuid4().hex[:24]
        expires_at = now_shanghai() + timedelta(
            hours=settings.upload_session_expire_hours
        )

        obj = create_upload_session(
            db=db,
            upload_uid=upload_uid,
            project_uid=req.project_uid,
            filename=filename,
            file_md5=req.file_md5,
            file_size=req.file_size,
            content_type=req.content_type,
            chunk_size=chunk_size,
            total_parts=total_parts,
            expires_at=expires_at,
            owner_id=user.id if user else None,
        )

        # 初始化 Redis Bitmap。这里失败不能影响主流程。
        self._init_upload_bitmap(upload_uid=obj.upload_uid)

        return InitUploadResponse(
            upload_uid=obj.upload_uid,
            filename=obj.filename,
            file_size=obj.file_size,
            file_md5=obj.file_md5,
            chunk_size=obj.chunk_size,
            total_parts=obj.total_parts,
            uploaded_parts=[],
            missing_parts=list(range(obj.total_parts)),
            status=obj.status,
        )

    def _bitmap_key(self, upload_uid: str) -> str:
        return f"upload:bitmap:{upload_uid}"

    def _init_upload_bitmap(self, upload_uid: str) -> None:
        key = self._bitmap_key(upload_uid)
        try:
            redis_client.delete(key)
            redis_client.setbit(key, 0, 0)
            redis_client.expire(key, settings.upload_session_expire_hours * 3600)
        except Exception:
            # Redis 是加速层，不能影响上传会话创建
            pass

    def _get_uploaded_parts_from_redis(
        self,
        upload_uid: str,
        total_parts: int,
    ) -> list[int]:
        key = self._bitmap_key(upload_uid)
        uploaded_parts: list[int] = []

        try:
            for i in range(total_parts):
                if redis_client.getbit(key, i):
                    uploaded_parts.append(i)
        except Exception:
            # Redis 失败时，第一轮先返回空。
            # 第二轮 upload_part 做完后，我们会从 MySQL upload_part 兜底恢复。
            return []

        return uploaded_parts

    async def upload_part(
        self,
        db: Session,
        upload_uid: str,
        part_number: int,
        part_md5: str,
        file_part: UploadFile,
    ) -> UploadPartResponse:
        session = get_upload_session_by_uid(db, upload_uid)
        if session is None:
            raise ValueError("上传会话不存在")

        if session.status == "completed":
            success_parts = list_success_part_numbers(db, upload_uid)

            uploaded_parts = self._merge_uploaded_parts(
                upload_uid=upload_uid,
                total_parts=session.total_parts,
                db_parts=success_parts,
            )

            # completed 状态下，即使 Redis 丢了，也应认为所有分片已完成
            if not uploaded_parts:
                uploaded_parts = list(range(session.total_parts))

            return UploadPartResponse(
                upload_uid=upload_uid,
                part_number=part_number,
                part_size=0,
                part_md5=part_md5,
                uploaded_parts=uploaded_parts,
                missing_parts=[],
                status=session.status,
                reused=True,
                message="文件已完成上传，忽略重复分片上传",
            )

        if session.status != UPLOAD_STATUS_UPLOADING:
            raise ValueError(f"当前上传状态不允许上传分片：{session.status}")

        if part_number < 0 or part_number >= session.total_parts:
            raise ValueError("part_number 超出范围")

        if not part_md5:
            raise ValueError("part_md5 不能为空")

        object_key = f"tmp/{upload_uid}/part-{part_number}"

        # Redis 分片级幂等锁：防止同一分片被并发重复上传。
        # Redis 是加速层，不可用时降级为不锁，不影响主流程。
        lock_key = f"upload:part:lock:{upload_uid}:{part_number}"
        lock_acquired = False
        try:
            lock_acquired = bool(redis_client.set(lock_key, "1", nx=True, ex=30))
        except Exception:
            lock_acquired = True  # Redis 不可用时跳过锁
        if not lock_acquired:
            raise ValueError(
                f"该分片正在上传中，请稍后重试，part_number={part_number}"
            )

        tmp_path = None
        part_size = 0
        md5 = hashlib.md5()

        try:
            # 写临时文件，避免一次性读入内存
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name

                while True:
                    chunk = await file_part.read(1024 * 1024)  # 每次读 1MB
                    if not chunk:
                        break

                    part_size += len(chunk)
                    md5.update(chunk)
                    tmp.write(chunk)

            real_md5 = md5.hexdigest()

            if real_md5 != part_md5:
                raise ValueError(
                    f"分片 MD5 校验失败，part_number={part_number}, "
                    f"client_md5={part_md5}, server_md5={real_md5}"
                )

            # 上传到 MinIO。MinIO 成功后，才写 MySQL 和 Redis。
            minio_storage.upload_file(
                object_key=object_key,
                file_path=tmp_path,
                content_type=file_part.content_type,
            )

            create_or_update_upload_part(
                db=db,
                upload_uid=upload_uid,
                part_number=part_number,
                part_size=part_size,
                part_md5=real_md5,
                object_key=object_key,
            )

            # MySQL 是事实兜底，Redis Bitmap 是快速状态
            try:
                redis_client.setbit(self._bitmap_key(upload_uid), part_number, 1)
                redis_client.expire(
                    self._bitmap_key(upload_uid),
                    settings.upload_session_expire_hours * 3600,
                )
            except Exception:
                # Redis 失败不能影响分片上传成功，因为 MySQL upload_part 已经记录成功
                pass

            success_parts = list_success_part_numbers(db, upload_uid)
            update_uploaded_parts_count(
                db=db,
                upload_uid=upload_uid,
                uploaded_parts=len(success_parts),
            )

            uploaded_parts = self._merge_uploaded_parts(
                upload_uid=upload_uid,
                total_parts=session.total_parts,
                db_parts=success_parts,
            )
            missing_parts = [
                i for i in range(session.total_parts)
                if i not in set(uploaded_parts)
            ]

            return UploadPartResponse(
                upload_uid=upload_uid,
                part_number=part_number,
                part_size=part_size,
                part_md5=real_md5,
                uploaded_parts=uploaded_parts,
                missing_parts=missing_parts,
                status=session.status,
            )

        finally:
            # 释放分片幂等锁
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass

            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _merge_uploaded_parts(
        self,
        upload_uid: str,
        total_parts: int,
        db_parts: list[int],
    ) -> list[int]:
        """
        合并 MySQL 和 Redis 的分片状态。
        MySQL 是可靠兜底，Redis 是快速状态。
        """
        result = set(db_parts)

        try:
            key = self._bitmap_key(upload_uid)
            for i in range(total_parts):
                if redis_client.getbit(key, i):
                    result.add(i)
        except Exception:
            pass

        return sorted(result)

    def get_upload_status(self, db: Session, upload_uid: str) -> UploadStatusResponse:
        session = get_upload_session_by_uid(db, upload_uid)
        if session is None:
            raise ValueError("上传会话不存在")

        success_parts = list_success_parts(db, upload_uid)

        # 以 MySQL upload_part 记录为事实源。
        # upload_part 流程已保证：先校验 MD5 → 上传 MinIO 成功 → 才写 MySQL。
        # MinIO 对象丢失是极小概率事件，不做逐分片 stat 调用避免高并发下的性能问题。
        uploaded_parts_set = {p.part_number for p in success_parts}

        uploaded_parts = sorted(uploaded_parts_set)

        # 用可靠结果修复 Redis Bitmap
        self._repair_upload_bitmap(
            upload_uid=upload_uid,
            total_parts=session.total_parts,
            uploaded_parts=uploaded_parts,
        )

        missing_parts = [
            i for i in range(session.total_parts)
            if i not in uploaded_parts_set
        ]

        return UploadStatusResponse(
            upload_uid=session.upload_uid,
            filename=session.filename,
            file_size=session.file_size,
            file_md5=session.file_md5,
            chunk_size=session.chunk_size,
            total_parts=session.total_parts,
            uploaded_parts=uploaded_parts,
            missing_parts=missing_parts,
            status=session.status,
            document_uid=session.document_uid,
            last_error=session.last_error,
        )

    def _repair_upload_bitmap(
        self,
        upload_uid: str,
        total_parts: int,
        uploaded_parts: list[int],
    ) -> None:
        """根据 MySQL 分片记录修复 Redis Bitmap。Redis 是加速层，不是事实源。"""
        key = self._bitmap_key(upload_uid)

        try:
            redis_client.delete(key)

            for part_number in uploaded_parts:
                redis_client.setbit(key, part_number, 1)

            redis_client.expire(
                key,
                settings.upload_session_expire_hours * 3600,
            )
        except Exception:
            # Redis 修复失败不影响状态查询结果
            pass

    def complete_upload(
        self,
        db: Session,
        req: CompleteUploadRequest,
        user=None,
    ) -> CompleteUploadResponse:
        session = get_upload_session_by_uid(db, req.upload_uid)
        if session is None:
            raise ValueError("上传会话不存在")

        # 如果已经完成，重复 complete 直接返回已有 document_uid
        if session.status == "completed" and session.document_uid:
            document_uid = session.document_uid
            parse_task = get_parse_task_by_document_uid(db, document_uid)

            return CompleteUploadResponse(
                upload_uid=session.upload_uid,
                document_uid=document_uid,
                filename=session.filename,
                file_size=session.file_size,
                file_md5=session.file_md5,
                object_key=f"documents/{session.project_uid}/{document_uid}/{session.filename}",
                parse_task_uid=parse_task.task_uid if parse_task else "",
                status=session.status,
                reused=True,
                message="文件已合并完成，复用已有文档",
            )

        # 先复用 upload_status 逻辑，确认分片完整
        status = self.get_upload_status(db=db, upload_uid=req.upload_uid)
        if status.missing_parts:
            raise ValueError(f"仍有分片未上传：{status.missing_parts}")

        # CAS 抢占合并权，避免重复 complete 并发合并
        locked = mark_upload_merging(db=db, upload_uid=req.upload_uid)
        if not locked:
            raise ValueError("当前上传会话正在合并或已完成，请勿重复提交")

        tmp_dir = tempfile.mkdtemp(prefix="complete_upload_")
        merged_path = os.path.join(tmp_dir, "merged_file")

        document_uid = "doc_" + uuid.uuid4().hex[:24]
        final_object_key = f"documents/{session.project_uid}/{document_uid}/{session.filename}"
        parse_task_uid = "parse_" + uuid.uuid4().hex[:24]

        # 提前持久化 document_uid 到 session，防止合并中途崩溃后最终文件成为孤儿
        session.document_uid = document_uid
        db.commit()

        try:
            # 重新查询 part 记录，按 part_number 顺序合并
            parts = list_success_parts(db, session.upload_uid)
            parts = sorted(parts, key=lambda p: p.part_number)

            if len(parts) != session.total_parts:
                raise ValueError("分片数量不完整，不能合并")

            md5 = hashlib.md5()

            with open(merged_path, "wb") as merged_file:
                for part in parts:
                    if not minio_storage.object_exists(part.object_key):
                        raise ValueError(f"分片对象不存在：part_number={part.part_number}")

                    part_tmp_path = os.path.join(tmp_dir, f"part-{part.part_number}")
                    minio_storage.download_file(part.object_key, part_tmp_path)

                    with open(part_tmp_path, "rb") as pf:
                        while True:
                            chunk = pf.read(1024 * 1024)
                            if not chunk:
                                break
                            md5.update(chunk)
                            merged_file.write(chunk)

            real_file_md5 = md5.hexdigest()
            if real_file_md5 != session.file_md5:
                raise ValueError(
                    f"完整文件 MD5 校验失败，client_md5={session.file_md5}, "
                    f"server_md5={real_file_md5}"
                )

            minio_storage.upload_file(
                object_key=final_object_key,
                file_path=merged_path,
                content_type=session.content_type,
            )

            document = create_document(
                db=db,
                document_uid=document_uid,
                project_uid=session.project_uid,
                filename=session.filename,
                file_md5=session.file_md5,
                file_size=session.file_size,
                file_type=os.path.splitext(session.filename)[1].lower().lstrip("."),
                bucket=settings.minio_bucket_documents,
                object_key=final_object_key,
                owner_id=user.id if user else None,
            )

            parse_task = create_document_parse_task(
                db=db,
                task_uid=parse_task_uid,
                document_uid=document.document_uid,
            )

            # Kafka 发送失败不应该让 document 和 parse_task 丢失。
            # 第一版：记录错误，但 complete 仍返回成功。
            try:
                send_document_parse_message(
                    {
                        "task_uid": parse_task.task_uid,
                        "document_uid": document.document_uid,
                        "object_key": final_object_key,
                    }
                )
            except Exception as exc:
                # 这里暂时只记录到 upload_session.last_error。
                # 后续会做 parse_task pending 补偿重投递。
                session.last_error = f"Kafka parse message send failed: {exc}"
                db.commit()

            mark_upload_completed(
                db=db,
                upload_uid=session.upload_uid,
                document_uid=document.document_uid,
            )

            # 合并完成后清理 MinIO 临时分片，best-effort，不影响返回
            self._cleanup_temp_parts(session.upload_uid)

            return CompleteUploadResponse(
                upload_uid=session.upload_uid,
                document_uid=document.document_uid,
                filename=session.filename,
                file_size=session.file_size,
                file_md5=session.file_md5,
                object_key=final_object_key,
                parse_task_uid=parse_task.task_uid,
                status="completed",
            )

        except Exception as exc:
            mark_upload_failed(db=db, upload_uid=session.upload_uid, error=str(exc))
            raise

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _cleanup_temp_parts(self, upload_uid: str) -> None:
        """best-effort 清理 MinIO 中临时分片对象。"""
        prefix = f"tmp/{upload_uid}/"
        try:
            deleted = minio_storage.remove_objects_by_prefix(prefix)
            if deleted > 0:
                import logging
                logging.getLogger(__name__).info(
                    "cleaned up %d temp parts for upload=%s", deleted, upload_uid
                )
        except Exception:
            pass

    def cleanup_expired_sessions(self, db: Session) -> int:
        """
        清理过期的上传会话：
        1. 删除 MinIO 中的临时分片对象 (tmp/{upload_uid}/)
        2. 如果会话有 document_uid（合并中途崩溃），同时清理孤儿最终文件
        3. 将会话状态标记为 expired
        返回清理的会话数量。
        """
        expired_sessions = get_expired_upload_sessions(db)
        cleaned = 0

        for session in expired_sessions:
            try:
                # 清理临时分片
                prefix = f"tmp/{session.upload_uid}/"
                minio_storage.remove_objects_by_prefix(prefix)

                # 清理孤儿最终文件（合并中途崩溃产生）
                if session.document_uid:
                    final_prefix = (
                        f"documents/{session.project_uid}/{session.document_uid}/"
                    )
                    minio_storage.remove_objects_by_prefix(final_prefix)

                mark_upload_expired(db, session.upload_uid)
                cleaned += 1
            except Exception:
                pass

        return cleaned

    def delete_document(self, db: Session, document_uid: str, user=None) -> dict:
        """级联删除文档：MySQL chunks → ES → FAISS → MinIO → MySQL document。"""
        doc = get_document_by_uid(db, document_uid)
        if not doc:
            raise ValueError("文档不存在")

        # 权限检查：仅文档所有者或管理员可删除（含历史 NULL owner 数据）
        if user and doc.owner_id != user.id:
            if getattr(user, "role", "") != "admin":
                raise ValueError("无权删除此文档")

        from app.models.document_chunk import DocumentChunk
        from app.services.vector_store import VectorStore
        from app.services.es_service import ESService

        # 1. 收集 chunk_uids，用于清理 FAISS
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_uid == document_uid)
            .all()
        )
        chunk_uids = [c.chunk_uid for c in chunks]

        # 2. 删除 ES 中的 chunk 文档
        try:
            es = ESService()
            es.delete_by_document_uid(document_uid)
        except Exception:
            pass

        # 3. 删除 FAISS 中的向量
        if chunk_uids:
            try:
                vs = VectorStore()
                vs.remove_by_chunk_uids(chunk_uids)
            except Exception:
                pass

        # 4. 删除 MySQL chunks
        delete_chunks_by_document_uid(db, document_uid)

        # 5. 删除 MinIO 最终文件
        try:
            minio_storage.remove_object(doc.object_key)
        except Exception:
            pass

        # 6. 清理上传会话中的临时分片（如有）
        try:
            upload_sessions = (
                db.query(UploadSession)
                .filter(UploadSession.document_uid == document_uid)
                .all()
            )
            for s in upload_sessions:
                prefix = f"tmp/{s.upload_uid}/"
                minio_storage.remove_objects_by_prefix(prefix)
        except Exception:
            pass

        # 7. 删除 MySQL document 记录
        from app.repository.document_repo import delete_document as repo_delete
        repo_delete(db, document_uid)

        return {"document_uid": document_uid, "filename": doc.filename, "chunks_removed": len(chunk_uids)}

    def cancel_upload(self, db: Session, upload_uid: str, user=None) -> dict:
        """取消上传：删除 MinIO 临时分片、关联文档及 chunks/ES/FAISS、上传会话。"""
        session = get_upload_session_by_uid(db, upload_uid)
        if session is None:
            raise ValueError("上传会话不存在")

        # 权限检查
        if user and session.owner_id is not None and session.owner_id != user.id:
            if getattr(user, "role", "") != "admin":
                raise ValueError("无权取消此上传")

        result = {"upload_uid": upload_uid, "document_uid": None, "cleaned": []}

        # 1. 删除关联文档（级联 chunks + ES + FAISS + MinIO）
        if session.document_uid:
            try:
                self.delete_document(db, session.document_uid)
                result["document_uid"] = session.document_uid
                result["cleaned"].append("document")
            except ValueError:
                pass

        # 2. 删除 MinIO 临时分片
        try:
            prefix = f"tmp/{upload_uid}/"
            minio_storage.remove_objects_by_prefix(prefix)
            result["cleaned"].append("minio_parts")
        except Exception:
            pass

        # 3. 删除 UploadPart 记录
        from app.models.upload_part import UploadPart
        deleted_parts = (
            db.query(UploadPart)
            .filter(UploadPart.upload_uid == upload_uid)
            .delete(synchronize_session=False)
        )
        result["deleted_parts"] = deleted_parts

        # 4. 删除上传会话
        db.delete(session)
        db.commit()
        result["cleaned"].append("session")

        return result

    def get_document_status(self, db: Session, document_uid: str, user=None) -> DocumentStatusResponse:
        doc = get_document_by_uid(db, document_uid)
        if doc is None:
            raise ValueError("文档不存在")

        # 权限检查：仅文档所有者或管理员可查看（含历史 NULL owner 数据）
        if user and doc.owner_id != user.id:
            if getattr(user, "role", "") != "admin":
                raise ValueError("无权查看此文档")

        return DocumentStatusResponse(
            document_uid=doc.document_uid,
            filename=doc.filename,
            file_size=doc.file_size,
            upload_status=doc.upload_status,
            parse_status=doc.parse_status,
            chunk_status=doc.chunk_status,
            embedding_status=doc.embedding_status,
            index_status=doc.index_status,
            available_for_search=doc.available_for_search,
            last_error=doc.last_error,
        )

    def reset_document_parse_status(db, document_uid: str):
        doc = db.query(Document).filter(
            Document.document_uid == document_uid
        ).first()

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