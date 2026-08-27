import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.database import SessionLocal
from app.services.upload_service import UploadService

logger = logging.getLogger(__name__)

_scheduler = None


def cleanup_expired_uploads():
    db = SessionLocal()
    try:
        service = UploadService()
        cleaned = service.cleanup_expired_sessions(db)
        if cleaned > 0:
            logger.info("清理过期上传会话完成 | cleaned=%s", cleaned)
    except Exception:
        logger.exception("清理过期上传会话失败")
    finally:
        db.close()


def start_upload_cleanup_scheduler():
    global _scheduler

    if _scheduler is not None:
        logger.info("上传会话清理定时任务已启动，跳过重复启动")
        return

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    scheduler.add_job(
        cleanup_expired_uploads,
        trigger="interval",
        minutes=30,
        id="upload_cleanup_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    _scheduler = scheduler

    logger.info("上传会话清理定时任务已启动 | interval=30min")


def stop_upload_cleanup_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Upload cleanup scheduler stopped")
