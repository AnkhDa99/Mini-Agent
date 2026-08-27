import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.database import SessionLocal
from app.core.redis_client import redis_client
from app.repository.cache_task_repo import (
    list_retryable_tasks,
    mark_task_success,
    mark_task_retry,
    mark_task_failed,
)

logger = logging.getLogger(__name__)


def retry_cache_invalidation_tasks():
    db = SessionLocal()

    try:
        tasks = list_retryable_tasks(db, limit=50)

        if not tasks:
            return

        logger.info("开始补偿缓存删除任务 | count=%s", len(tasks))

        for task in tasks:
            try:
                redis_client.delete(task.cache_key)

                mark_task_success(db, task.task_uid)

                logger.info(
                    "缓存删除补偿成功 | task_uid=%s cache_key=%s",
                    task.task_uid,
                    task.cache_key,
                )

            except Exception as exc:
                next_retry_count = task.retry_count + 1

                if next_retry_count >= task.max_retry:
                    mark_task_failed(db, task.task_uid, str(exc))

                    logger.error(
                        "缓存删除补偿最终失败 | task_uid=%s retry_count=%s max_retry=%s error=%s",
                        task.task_uid,
                        next_retry_count,
                        task.max_retry,
                        exc,
                    )
                else:
                    mark_task_retry(
                        db=db,
                        task_uid=task.task_uid,
                        error=str(exc),
                        delay_seconds=30,
                    )

                    logger.warning(
                        "缓存删除补偿失败，等待下次重试 | task_uid=%s retry_count=%s error=%s",
                        task.task_uid,
                        next_retry_count,
                        exc,
                    )

    finally:
        db.close()


def start_cache_retry_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    scheduler.add_job(
        retry_cache_invalidation_tasks,
        trigger="interval",
        seconds=30,
        id="cache_invalidation_retry_job",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()

    logger.info("缓存删除补偿定时任务已启动")