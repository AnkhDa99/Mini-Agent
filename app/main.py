import logging

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.database import Base, engine, SessionLocal
from app.core.config import get_settings
from app.core.logger import setup_logger
from app.core import auth as auth_utils

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.cache_invalidation_task import CacheInvalidationTask
from app.models.upload_session import UploadSession
from app.models.upload_part import UploadPart
from app.models.document import Document
from app.models.document_parse_task import DocumentParseTask
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.models.scenario import (
    ScenarioTemplate, KnowledgeEntry, ScenarioMatchLog, KnowledgeRevision,
)

from app.services.cache_retry_scheduler import start_cache_retry_scheduler, stop_cache_retry_scheduler
from app.services.upload_cleanup_scheduler import start_upload_cleanup_scheduler, stop_upload_cleanup_scheduler
from app.services.embedding_worker import start_embedding_worker, stop_embedding_worker
from app.services.parse_worker import start_parse_worker, stop_parse_worker
from app.mq.document_parse_consumer import start_document_parse_consumer, stop_document_parse_consumer

from app.api.routes.chat import router as chat_router
from app.api.routes.document import router as document_router
from app.api.routes.auth import router as auth_router
from app.api.routes.admin import router as admin_router
from app.api.routes.scenario import router as scenario_router

setup_logger()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(scenario_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

    # 迁移：为已有 message 表添加 sources_json 列
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE message ADD COLUMN sources_json TEXT NULL"))
            conn.commit()
            logger.info("Migration: added sources_json column to message table")
        except Exception:
            conn.rollback()
            pass  # 列已存在则跳过

        # Issue 2: 为已有 message 表添加 metadata_json 列（source usage metadata）
        try:
            conn.execute(text("ALTER TABLE message ADD COLUMN metadata_json TEXT NULL"))
            conn.commit()
            logger.info("Migration: added metadata_json column to message table")
        except Exception:
            conn.rollback()
            pass  # 列已存在则跳过

    # 首次启动自动创建管理员
    from app.repository.user_repo import count_users, create_user
    db = SessionLocal()
    try:
        if count_users(db) == 0 and settings.initial_admin_password:
            create_user(
                db,
                settings.initial_admin_username,
                auth_utils.hash_password(settings.initial_admin_password),
                role="admin",
            )
            logger.info("Initial admin user created: %s", settings.initial_admin_username)
    finally:
        db.close()

    # 初始化 Neo4j 约束和索引
    try:
        from app.core.neo4j import init_neo4j_constraints
        init_neo4j_constraints()
        logger.info("Neo4j constraints initialized")
    except Exception:
        logger.warning("Neo4j initialization failed — graph features will be unavailable")

    start_cache_retry_scheduler()
    start_upload_cleanup_scheduler()
    start_embedding_worker()
    start_document_parse_consumer()  # MQ 主路径
    start_parse_worker()            # APScheduler 兜底（MQ 不可用时）


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down background services...")
    # APScheduler 必须先停，否则线程池先销毁会导致 "cannot schedule new futures after shutdown"
    stop_parse_worker()
    stop_embedding_worker()
    stop_cache_retry_scheduler()
    stop_upload_cleanup_scheduler()
    # Kafka consumer daemon thread 最后停
    stop_document_parse_consumer()
    logger.info("All background services stopped")


templates = Jinja2Templates(directory=settings.templates_dir)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "app_name": settings.app_name, "contact_admin_email": settings.contact_admin_email},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "app_name": settings.app_name, "contact_admin_email": settings.contact_admin_email},
    )


@app.get("/api/documents/download/{filename:path}")
async def download_document(filename: str):
    """下载生成的文档（PPT/Word/PDF/Excel）。"""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(settings.document_output_dir, safe_name)
    if not os.path.isfile(file_path):
        raise StarletteHTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=file_path,
        filename=safe_name,
        media_type="application/octet-stream",
    )
