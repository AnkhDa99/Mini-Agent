"""场景匹配引擎 — FAISS 语义匹配 + Neo4j 图谱扩展。

流程：
1. 用户 query → embedding
2. FAISS 语义匹配 knowledge_entry（filter: status=approved）
3. Neo4j 图谱扩展：通过关联关系召回更多相关条目
4. 按 similarity 排序，threshold 过滤，返回 top_k
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from app.core.config import settings
from app.core.database import SessionLocal
from app.repository import scenario_repo as repo
from app.repository.scenario_repo import increment_entry_usage, create_match_log
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ScenarioMatcher:
    """场景匹配器 — 管理 FAISS 索引的生命周期和查询。"""

    def __init__(self):
        self.dim = settings.embedding_dim
        self.index_path = Path(settings.scenario_faiss_index_path)
        self.index_file = self.index_path / "scenario_index.faiss"
        self.mapping_file = self.index_path / "scenario_mapping.json"

        # entry_uid → FAISS internal id
        self.id_to_entry: list[str] = []
        self.entry_to_id: dict[str, int] = {}

        self.index: faiss.Index | None = None
        self._embedding_service: EmbeddingService | None = None
        self._load_or_create()

    def _get_embedding(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _load_or_create(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self.index_file.exists() and self.mapping_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.id_to_entry = json.load(f)
                self.entry_to_id = {
                    uid: i for i, uid in enumerate(self.id_to_entry) if uid is not None
                }
                logger.info(
                    "Scenario FAISS loaded: %d entries, dim=%d",
                    self.index.ntotal, self.dim,
                )
                return
            except Exception:
                logger.exception("Failed to load scenario FAISS, creating new")

        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        self.id_to_entry = []
        self.entry_to_id = {}

    def index_entry(self, entry_uid: str, plain_text: str) -> bool:
        """将单个知识条目加入 FAISS 索引。已存在则更新。"""
        emb = self._get_embedding()
        vecs = emb.embed_texts([plain_text])
        if not vecs:
            return False

        vec = np.array([vecs[0]], dtype=np.float32)
        faiss.normalize_L2(vec)

        if entry_uid in self.entry_to_id:
            # 更新已有向量（IndexIDMap 不支持 update，需要移除再添加）
            old_id = self.entry_to_id[entry_uid]
            self.index.remove_ids(np.array([old_id], dtype=np.int64))
            self.id_to_entry[old_id] = None

        new_id = len(self.id_to_entry)
        self.index.add_with_ids(vec, np.array([new_id], dtype=np.int64))
        self.id_to_entry.append(entry_uid)
        self.entry_to_id[entry_uid] = new_id
        return True

    def remove_entry(self, entry_uid: str):
        """从索引中移除条目。"""
        if entry_uid not in self.entry_to_id:
            return
        old_id = self.entry_to_id.pop(entry_uid)
        self.index.remove_ids(np.array([old_id], dtype=np.int64))
        self.id_to_entry[old_id] = None

    def rebuild_index(self):
        """全量重建索引：从 MySQL 读取所有 approved 且模板启用的条目，重新建 FAISS。"""
        db = SessionLocal()
        try:
            all_entries = repo.list_entries(
                db, status="approved", offset=0, limit=100_000,
            )
            # 过滤掉所属模板已停用的条目
            entries = []
            for e in all_entries:
                tmpl = repo.get_template_by_uid(db, e.template_uid)
                if tmpl and not tmpl.is_active:
                    logger.debug("Skipping entry %s (template %s is inactive)", e.entry_uid, e.template_uid)
                    continue
                entries.append(e)
        finally:
            db.close()

        # 重建
        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        self.id_to_entry = []
        self.entry_to_id = {}

        if not entries:
            logger.info("Scenario index rebuilt: 0 entries")
            self._save()
            return

        texts = [e.plain_text for e in entries]
        emb = self._get_embedding()
        vecs = emb.embed_texts(texts)
        if not vecs or len(vecs) != len(texts):
            logger.error("Embedding failed during rebuild")
            return

        np_vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(np_vecs)

        ids = np.arange(len(entries), dtype=np.int64)
        self.index.add_with_ids(np_vecs, ids)
        self.id_to_entry = [e.entry_uid for e in entries]
        self.entry_to_id = {e.entry_uid: i for i, e in enumerate(entries)}

        self._save()
        logger.info("Scenario index rebuilt: %d entries", len(entries))

    def match(
        self,
        db,
        query: str,
        user=None,
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> dict:
        """场景匹配主入口。返回匹配结果字典。"""
        if not settings.scenario_kb_enabled:
            return {"query": query, "entries": [], "match_count": 0, "elapsed_ms": 0}

        start = time.perf_counter()

        # 1. Query embedding
        emb = self._get_embedding()
        q_vecs = emb.embed_texts([query])
        if not q_vecs or self.index.ntotal == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return {"query": query, "entries": [], "match_count": 0, "elapsed_ms": elapsed}

        q_vec = np.array([q_vecs[0]], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        # 2. FAISS 搜索
        k = min(top_k * 3, self.index.ntotal)  # 多召回一些用于阈值过滤和图谱扩展
        scores, ids = self.index.search(q_vec, k)

        # 诊断：记录 top-3 分数（不管是否超过阈值）
        top3_scores = [float(s) for s in scores[0][:3] if float(s) > 0]
        top3_uids = [self.id_to_entry[i] if 0 <= i < len(self.id_to_entry) else None for i in ids[0][:3]]
        logger.info(
            "Scenario FAISS raw top-3 | q=%.40s | scores=%s | uids=%s",
            query, [round(s, 4) for s in top3_scores], top3_uids,
        )

        # 3. 收集候选条目
        candidates: list[dict] = []
        matched_template_uid = None

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.id_to_entry):
                continue
            entry_uid = self.id_to_entry[idx]
            if entry_uid is None:
                continue

            similarity = float(score)
            if similarity < threshold:
                continue

            entry = repo.get_entry_by_uid(db, entry_uid)
            if not entry or entry.status != "approved":
                continue

            tmpl = repo.get_template_by_uid(db, entry.template_uid)
            if tmpl and not tmpl.is_active:
                continue  # 模板已停用，跳过

            candidates.append({
                "entry_uid": entry.entry_uid,
                "title": entry.title,
                "content_json": entry.content_json,
                "template_name": tmpl.name if tmpl else "",
                "template_icon": tmpl.icon if tmpl else "",
                "similarity_score": round(similarity, 4),
                "tags": entry.tags or "",
            })
            if matched_template_uid is None:
                matched_template_uid = entry.template_uid

        # 4. 图谱扩展：通过 Neo4j 关联关系召回更多条目
        if candidates:
            matched_uids = [c["entry_uid"] for c in candidates]
            from app.core.neo4j import get_expanded_match
            expanded_uids = get_expanded_match(matched_uids, max_related=top_k)
            existing_uids = set(matched_uids)
            for uid in expanded_uids:
                if uid in existing_uids:
                    continue
                entry = repo.get_entry_by_uid(db, uid)
                if not entry or entry.status != "approved":
                    continue
                tmpl = repo.get_template_by_uid(db, entry.template_uid)
                if tmpl and not tmpl.is_active:
                    continue
                candidates.append({
                    "entry_uid": entry.entry_uid,
                    "title": entry.title,
                    "content_json": entry.content_json,
                    "template_name": tmpl.name if tmpl else "",
                    "template_icon": tmpl.icon if tmpl else "",
                    "similarity_score": 0.5,  # 图谱扩展条目标记较低分数
                    "tags": entry.tags or "",
                })
                existing_uids.add(uid)

        # 5. 排序取 top_k
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_entries = candidates[:top_k]

        # 6. 记录匹配日志 + 更新使用计数
        if final_entries:
            matched_uids = [e["entry_uid"] for e in final_entries]
            max_score = final_entries[0]["similarity_score"]
            try:
                create_match_log(
                    db,
                    query_text=query,
                    matched_template_uid=matched_template_uid,
                    matched_entry_uids=json.dumps(matched_uids, ensure_ascii=False),
                    similarity_score=max_score,
                    user_id=user.id if user else None,
                    conversation_uid=None,
                )
                for e in final_entries:
                    increment_entry_usage(db, e["entry_uid"])
            except Exception:
                logger.exception("Failed to log match")

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Scenario match | q=%.50s → %d/%d results in %.1fms (threshold=%.2f)",
            query, len(final_entries), self.index.ntotal, elapsed, threshold,
        )

        return {
            "query": query,
            "entries": final_entries,
            "match_count": len(final_entries),
            "matched_template_uid": matched_template_uid,
            "elapsed_ms": round(elapsed, 1),
        }

    def _save(self):
        """持久化 FAISS 索引和映射文件。"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.id_to_entry, f, ensure_ascii=False)


# 全局单例
_matcher: ScenarioMatcher | None = None


def get_scenario_matcher() -> ScenarioMatcher:
    global _matcher
    if _matcher is None:
        _matcher = ScenarioMatcher()
    return _matcher
