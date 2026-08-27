import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS 本地向量索引。
    使用 IndexIDMap(IndexFlatIP) ，内积 = 余弦相似度（向量已归一化）。
    支持磁盘持久化：index.faiss + mapping.json。
    """

    def __init__(self):
        self.dim = settings.embedding_dim
        self.index_path = Path(settings.faiss_index_path)
        self.index_file = self.index_path / "index.faiss"
        self.mapping_file = self.index_path / "mapping.json"

        # chunk_uid → internal FAISS id 的双向映射
        # id_to_chunk: list，索引即 FAISS internal id
        self.id_to_chunk: list[str] = []
        # chunk_to_id: dict，用于去重检查
        self.chunk_to_id: dict[str, int] = {}

        self.index: faiss.Index | None = None
        self._load_or_create()

    def _load_or_create(self):
        self.index_path.mkdir(parents=True, exist_ok=True)

        if self.index_file.exists() and self.mapping_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.id_to_chunk = json.load(f)
                self.chunk_to_id = {
                    cuid: i for i, cuid in enumerate(self.id_to_chunk) if cuid is not None
                }
                logger.info(
                    "FAISS index loaded: %d vectors, dim=%d",
                    self.index.ntotal,
                    self.dim,
                )
                return
            except Exception:
                logger.exception("Failed to load FAISS index, creating new one")

        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        self.id_to_chunk = []
        self.chunk_to_id = {}

    def add(self, chunk_uids: list[str], vectors: list[list[float]]) -> int:
        """批量添加向量。已存在的 chunk_uid 会被跳过。返回新增数量。"""
        if not chunk_uids or not vectors:
            return 0

        new_uids = []
        new_vecs = []
        start_id = len(self.id_to_chunk)

        for cuid, vec in zip(chunk_uids, vectors):
            if cuid in self.chunk_to_id:
                continue
            new_uids.append(cuid)
            new_vecs.append(vec)

        if not new_vecs:
            return 0

        np_vecs = np.array(new_vecs, dtype=np.float32)
        # 归一化，使内积等价于余弦相似度
        faiss.normalize_L2(np_vecs)

        ids = np.arange(start_id, start_id + len(new_vecs), dtype=np.int64)
        self.index.add_with_ids(np_vecs, ids)

        for i, cuid in enumerate(new_uids):
            faiss_id = start_id + i
            self.id_to_chunk.append(cuid)
            self.chunk_to_id[cuid] = faiss_id

        self._persist()
        return len(new_vecs)

    def search(self, query_vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        """搜索 top-k 相似向量，返回 [(chunk_uid, score), ...]。"""
        if self.index.ntotal == 0:
            return []

        np_query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(np_query)

        scores, ids = self.index.search(np_query, min(k, self.index.ntotal))

        results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:
                continue
            if 0 <= faiss_id < len(self.id_to_chunk):
                cuid = self.id_to_chunk[faiss_id]
                if cuid is not None:
                    results.append((cuid, float(score)))
        return results

    def remove_by_chunk_uids(self, chunk_uids: list[str]) -> int:
        """删除指定 chunk_uid 的向量。IndexIDMap 支持 remove_ids，无需重建。"""
        ids_to_remove = []
        for cuid in chunk_uids:
            if cuid in self.chunk_to_id:
                ids_to_remove.append(self.chunk_to_id[cuid])
                del self.chunk_to_id[cuid]

        if not ids_to_remove:
            return 0

        self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))

        # 将 id_to_chunk 中对应位置标记为 None（保留索引对齐，不重建）
        for faiss_id in ids_to_remove:
            if 0 <= faiss_id < len(self.id_to_chunk):
                self.id_to_chunk[faiss_id] = None

        self._persist()
        return len(ids_to_remove)

    def _persist(self):
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.mapping_file, "w", encoding="utf-8") as f:
                json.dump(self.id_to_chunk, f, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to persist FAISS index")

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
