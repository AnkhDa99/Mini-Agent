import os
import tempfile

from sqlalchemy.orm import Session
from app.core.config import settings

# from app.services.document_extractors import DocumentExtractor

from app.services.tika_extractor import TikaExtractor

from app.repository.document_repo import (
    get_document_by_uid,
    update_document_parse_success,
    update_document_parse_failed,
)
from app.repository.chunk_repo import bulk_create_chunks
from app.repository.parse_task_repo import (
    mark_parse_task_processing,
    mark_parse_task_success,
    mark_parse_task_failed,
    mark_parse_task_retry_or_failed,
)
from app.services.chunk_service import ChunkService
from app.storage.minio_client import minio_storage


class ParseService:
    def __init__(self):
        self.chunk_service = ChunkService()
        self.extractor = TikaExtractor()

    def parse_document(
        self,
        db: Session,
        task_uid: str,
        document_uid: str,
        worker_id: str,
    ):
        locked = mark_parse_task_processing(
            db=db,
            task_uid=task_uid,
            locked_by=worker_id,
        )

        if not locked:
            print(f"任务未抢占成功，跳过 task_uid={task_uid}")
            return

        doc = get_document_by_uid(db, document_uid)
        if doc is None:
            mark_parse_task_failed(db, task_uid, "document 不存在")
            return

        tmp_dir = tempfile.mkdtemp(prefix="parse_doc_")
        local_path = os.path.join(tmp_dir, doc.filename)

        try:
            minio_storage.download_file(doc.object_key, local_path)

            total_chunks = 0
            chunk_index_offset = 0

            for block in self.extractor.extract(local_path):
                parent_text = self._add_source_marker(block)

                chunks = self.chunk_service.split_text(
                    document_uid=doc.document_uid,
                    project_uid=doc.project_uid,
                    text=parent_text,
                    version=doc.current_version,
                    page_no=block.page_no,
                )

                for i, chunk in enumerate(chunks):
                    chunk.chunk_index = chunk_index_offset + i

                if chunks:
                    bulk_create_chunks(db, chunks)
                    total_chunks += len(chunks)
                    chunk_index_offset += len(chunks)

            if total_chunks == 0:
                raise ValueError(
                    "Tika 未提取到有效文本，可能是扫描版 PDF、图片型文档或文件内容为空"
                )

            update_document_parse_success(db, doc.document_uid)
            mark_parse_task_success(db, task_uid)

            print(
                f"解析完成 document_uid={doc.document_uid}, "
                f"chunks={total_chunks}"
            )


        except Exception as exc:

            error = str(exc)

            update_document_parse_failed(db, doc.document_uid, error)

            mark_parse_task_retry_or_failed(db, task_uid, error)

            raise

        finally:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _add_source_marker(self, block) -> str:
        markers = [f"[SOURCE_TYPE={getattr(block, 'source_type', 'unknown')}]"]

        page_no = getattr(block, "page_no", None)
        slide_no = getattr(block, "slide_no", None)
        sheet_name = getattr(block, "sheet_name", None)
        start_offset = getattr(block, "start_offset", None)
        end_offset = getattr(block, "end_offset", None)

        if page_no is not None:
            markers.append(f"[PAGE={page_no}]")

        if slide_no is not None:
            markers.append(f"[SLIDE={slide_no}]")

        if sheet_name is not None:
            markers.append(f"[SHEET={sheet_name}]")

        if start_offset is not None:
            markers.append(f"[START_OFFSET={start_offset}]")

        if end_offset is not None:
            markers.append(f"[END_OFFSET={end_offset}]")

        return "\n".join(markers) + "\n" + block.content