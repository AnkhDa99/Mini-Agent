from datetime import datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def now_shanghai() -> datetime:
    """
    返回中国时区的 naive datetime，方便写入 MySQL DATETIME 字段。
    """
    return datetime.now(SHANGHAI_TZ).replace(tzinfo=None)