from pydantic import BaseModel, Field


class InitUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., gt=0)
    file_md5: str = Field(..., min_length=32, max_length=64)
    content_type: str | None = None
    project_uid: str = "default_project"
    chunk_size: int | None = None


class InitUploadResponse(BaseModel):
    upload_uid: str
    filename: str
    file_size: int
    file_md5: str
    chunk_size: int
    total_parts: int
    uploaded_parts: list[int]
    missing_parts: list[int]
    status: str
    # 重复上传复用
    document_uid: str | None = None
    parse_task_uid: str | None = None
    reused: bool = False
    message: str | None = None

class UploadPartResponse(BaseModel):
    upload_uid: str
    part_number: int
    part_size: int
    part_md5: str
    uploaded_parts: list[int]
    missing_parts: list[int]
    status: str
    # completed 幂等兜底
    reused: bool = False
    message: str | None = None

class UploadStatusResponse(BaseModel):
    upload_uid: str
    filename: str
    file_size: int
    file_md5: str
    chunk_size: int
    total_parts: int
    uploaded_parts: list[int]
    missing_parts: list[int]
    status: str
    document_uid: str | None = None
    last_error: str | None = None

class CompleteUploadRequest(BaseModel):
    upload_uid: str

class CompleteUploadResponse(BaseModel):
    upload_uid: str
    document_uid: str
    filename: str
    file_size: int
    file_md5: str
    object_key: str
    parse_task_uid: str
    status: str
    reused: bool = False
    message: str | None = None

class DocumentStatusResponse(BaseModel):
    document_uid: str
    filename: str
    file_size: int
    upload_status: str
    parse_status: str
    chunk_status: str
    embedding_status: str
    index_status: str
    available_for_search: bool
    last_error: str | None = None