import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    向量化服务，兼容 OpenAI / SiliconFlow / DeepSeek 等 OpenAI 协议 API。
    优先级：embedding_api_key > openai_api_key（向后兼容）。
    """

    def __init__(self):
        api_key = settings.embedding_api_key or settings.openai_api_key
        base_url = settings.embedding_base_url or settings.openai_base_url or None

        # trust_env=False 防止读取 HTTP_PROXY/HTTPS_PROXY 环境变量，
        # DashScope 是阿里云国内服务，走代理反而导致 SSL 握手失败
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding，内部并发调用 API，返回与 texts 等长的向量列表。"""
        if not texts:
            return []

        batch_size = settings.embedding_batch_size  # API 单批限制（DashScope=10）
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i : i + batch_size]))

        if len(batches) == 1:
            return self._call_api(batches[0][1])

        # 并发 API 调用
        results = {}  # batch_index → embeddings
        max_workers = min(len(batches), 8)  # 最多 8 并发
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_api, batch): idx
                for idx, batch in batches
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Embedding batch failed, range start=%d", idx)

        # 按原始顺序组装结果；失败的批次抛异常
        all_embeddings = []
        for idx, _ in batches:
            if idx not in results:
                raise RuntimeError(
                    f"Embedding batch failed at offset {idx}, "
                    f"aborting to avoid misaligned results"
                )
            all_embeddings.extend(results[idx])

        return all_embeddings

    def _call_api(self, batch: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=batch,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, query: str) -> list[float]:
        """单条查询 embedding。"""
        resp = self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return resp.data[0].embedding
