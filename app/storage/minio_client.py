from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.error import S3Error
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

from app.core.config import settings


class MinioStorageClient:
    def __init__(self):
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket_documents
        self._ensure_bucket()
        self._ensure_lifecycle()

    def _ensure_bucket(self):
        try:
            exists = self.client.bucket_exists(self.bucket)
            if not exists:
                self.client.make_bucket(self.bucket)
        except S3Error as exc:
            raise RuntimeError(f"初始化 MinIO bucket 失败：{exc}") from exc

    def _ensure_lifecycle(self):
        """配置 bucket 生命周期：tmp/ 前缀对象在 2 天后自动过期删除。"""
        try:
            config = LifecycleConfig(
                [
                    Rule(
                        ENABLED,
                        rule_filter=Filter(prefix="tmp/"),
                        rule_id="expire-tmp-uploads",
                        expiration=Expiration(days=2),
                    ),
                ],
            )
            self.client.set_bucket_lifecycle(self.bucket, config)
        except S3Error:
            pass

    def upload_file(self, object_key: str, file_path: str, content_type: str | None = None):
        self.client.fput_object(
            bucket_name=self.bucket,
            object_name=object_key,
            file_path=file_path,
            content_type=content_type or "application/octet-stream",
        )

    def object_exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except Exception:
            return False

    def remove_object(self, object_key: str):
        try:
            self.client.remove_object(self.bucket, object_key)
        except Exception:
            pass

    def download_file(self, object_key: str, file_path: str):
        self.client.fget_object(
            bucket_name=self.bucket,
            object_name=object_key,
            file_path=file_path,
        )

    def remove_objects_by_prefix(self, prefix: str) -> int:
        """删除指定前缀下的所有对象，返回删除数量。best-effort，不抛异常。"""
        try:
            objects = list(
                self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            )
        except Exception:
            return 0

        if not objects:
            return 0

        try:
            errors = self.client.remove_objects(
                self.bucket,
                [obj.object_name for obj in objects],
            )
            error_count = sum(1 for _ in errors)
            return len(objects) - error_count
        except Exception:
            return 0


minio_storage = MinioStorageClient()