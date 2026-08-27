import logging

from elasticsearch import Elasticsearch

from app.core.config import settings

logger = logging.getLogger(__name__)


# 索引 mapping 版本号。修改 mapping 后递增此值，启动时自动重建索引 + 回填数据。
ES_MAPPING_VERSION = 6

# ── IK 同义词字典（项目专有术语）──
# 技术术语映射：搜索任何一个词，等同于搜索其同义词组
IK_SYNONYMS = [
    "FAISS, 向量检索, 向量搜索, 语义搜索 => faiss 向量检索",
    "ES, Elasticsearch, 关键词检索, 全文检索 => elasticsearch 关键词检索",
    "RRF, 融合排序, 混合检索 => rrf 融合排序",
    "RAG, 检索增强生成, 知识问答 => rag 检索增强生成",
    "Embedding, 向量化, 嵌入 => embedding 向量化",
    "chunk, 分块, 分段 => chunk 分块",
    "HyDE, 假设文档, 假想文档 => hyde 假设文档",
    "Reranker, 精排, 重排序 => reranker 精排",
    "MinIO, 对象存储 => minio 对象存储",
    "IOC, 控制反转, DI, 依赖注入 => ioc 控制反转",
    "Agent, 智能体 => agent 智能体",
    "MCP, 上下文协议 => mcp 上下文协议",
    "pipeline, 流水线 => pipeline 流水线",
]


def _check_ik_available(client: Elasticsearch) -> bool:
    """检测 ES 是否安装了 IK 分词器。"""
    try:
        resp = client.indices.analyze(
            body={"analyzer": "ik_smart", "text": "测试"}
        )
        return "error" not in resp
    except Exception:
        return False


class ESService:
    """
    Elasticsearch BM25 关键词检索服务。
    每个 chunk 作为一个 ES document，content 字段做全文检索。
    """

    def __init__(self):
        self.client = Elasticsearch(
            settings.es_host,
            request_timeout=10,
            retry_on_timeout=True,
        )
        self.index_name = settings.es_index_chunks
        self._needs_reindex = False
        self._ensure_index()

    def _ensure_index(self):
        """创建/升级索引。mapping 版本变更时自动重建（旧数据由 backfill 回填）。
        自动检测 IK 分词器：可用则 ik_max_word/ik_smart + 同义词过滤器，否则回退 ngram。
        字段权重：section_title^3 > heading_path^2 > content^2"""
        try:
            self.client.info()
        except Exception as e:
            raise ConnectionError(f"Elasticsearch 无法连接: {settings.es_host} — {e}")

        ik_available = _check_ik_available(self.client)
        analyzer_name = "ik_max_word" if ik_available else "ngram_analyzer"
        search_analyzer = "ik_smart" if ik_available else None

        self._current_analyzer = analyzer_name

        body: dict = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "b": 0.25,
                            "k1": 1.2,
                        }
                    },
                },
                "analysis": {},
            },
            "mappings": {
                "_meta": {
                    "version": ES_MAPPING_VERSION,
                    "analyzer": analyzer_name,
                    "synonyms_enabled": ik_available,
                },
                "properties": {
                    "chunk_uid": {"type": "keyword"},
                    "document_uid": {"type": "keyword"},
                    "project_uid": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content": {
                        "type": "text",
                        "analyzer": analyzer_name,
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 512},
                        },
                    },
                    "section_title": {
                        "type": "text",
                        "analyzer": analyzer_name,
                    },
                    "heading_path": {
                        "type": "text",
                        "analyzer": analyzer_name,
                    },
                    "block_type": {"type": "keyword"},
                    "page_no": {"type": "integer"},
                    "filename": {"type": "keyword"},
                }
            },
        }

        # IK 专用配置：同义词过滤器 + 搜索用粗粒度
        if ik_available:
            # 添加同义词过滤器
            body["settings"]["analysis"]["filter"] = {
                "ik_synonym_filter": {
                    "type": "synonym",
                    "synonyms": IK_SYNONYMS,
                    "expand": True,
                }
            }
            # 索引时：ik_max_word → synonym（细粒度 + 同义词扩展）
            # 搜索时：ik_smart（粗粒度，同义词在索引侧已处理）
            body["settings"]["analysis"]["analyzer"] = {
                "ik_synonym": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["ik_synonym_filter"],
                }
            }
            # 索引用 ik_synonym（含同义词），搜索用 ik_smart
            for field in ("content", "section_title", "heading_path"):
                body["mappings"]["properties"][field]["analyzer"] = "ik_synonym"
                body["mappings"]["properties"][field]["search_analyzer"] = search_analyzer

        else:
            # ngram 回退
            body["settings"]["analysis"]["tokenizer"] = {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 1,
                    "max_gram": 3,
                }
            }
            body["settings"]["analysis"]["analyzer"] = {
                "ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "ngram_tokenizer",
                }
            }
            body["settings"]["index"]["max_ngram_diff"] = 2

        # 检查已有索引是否需要升级
        if self.client.indices.exists(index=self.index_name):
            existing = self.client.indices.get(index=self.index_name)
            meta = existing[self.index_name].get("mappings", {}).get("_meta", {})
            existing_version = str(meta.get("version", 0))
            existing_analyzer = meta.get("analyzer", "")
            existing_synonyms = str(meta.get("synonyms_enabled", False))
            expected_synonyms = str(ik_available)
            if existing_version == str(ES_MAPPING_VERSION) and existing_analyzer == analyzer_name and existing_synonyms == expected_synonyms:
                return

            logger.warning(
                "ES mapping changed (%s v%s syn=%s → %s v%s syn=%s), deleting old index for rebuild",
                existing_analyzer, existing_version, existing_synonyms,
                analyzer_name, ES_MAPPING_VERSION, expected_synonyms,
            )
            self.client.indices.delete(index=self.index_name)

        try:
            self.client.indices.create(index=self.index_name, body=body)
            logger.info("ES index created (v%d, %s, synonyms=%s): %s",
                        ES_MAPPING_VERSION, analyzer_name, ik_available, self.index_name)
            self._needs_reindex = True
        except Exception:
            logger.exception("ES index creation failed — search will be FAISS-only")

    @property
    def needs_reindex(self) -> bool:
        """索引重建后需将 MySQL 中所有 chunk 的 index_status 重置，触发 backfill 回填 ES。"""
        return self._needs_reindex

    def index_chunks(self, chunks: list[dict]) -> int:
        """
        批量索引 chunks 到 ES。
        每个 dict 需包含: chunk_uid, document_uid, project_uid, chunk_index,
                       content, section_title, heading_path, block_type, page_no
        返回成功索引数量。
        """
        if not chunks:
            return 0

        from elasticsearch.helpers import bulk

        actions = []
        for chunk in chunks:
            actions.append({
                "_index": self.index_name,
                "_id": chunk["chunk_uid"],
                "_source": {
                    "chunk_uid": chunk["chunk_uid"],
                    "document_uid": chunk["document_uid"],
                    "project_uid": chunk.get("project_uid", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content": chunk["content"],
                    "section_title": chunk.get("section_title", ""),
                    "heading_path": chunk.get("heading_path", ""),
                    "block_type": chunk.get("block_type", ""),
                    "page_no": chunk.get("page_no", 0),
                    "filename": chunk.get("filename", ""),
                },
            })

        try:
            success, errors = bulk(self.client, actions, refresh=True)
            if errors:
                logger.warning("ES bulk index errors: %s", errors)
            return success
        except Exception:
            logger.exception("ES bulk index failed")
            return 0

    def search(self, query: str, project_uid: str = "", k: int = 20) -> list[dict]:
        """
        BM25 + 字段权重 + 同义词检索。
        字段权重：section_title^3 > heading_path^2 > content^2
        索引用 ik_synonym（ik_max_word + 同义词扩展），搜索用 ik_smart（粗粒度）。
        返回 [{chunk_uid, document_uid, content, score, ...}, ...]
        """
        def _do_search() -> list[dict]:
            body: dict = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "section_title^3",
                                        "heading_path^2",
                                        "content^2",
                                    ],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "minimum_should_match": "2<75%",
                                }
                            },
                        ],
                    }
                },
                "size": k,
            }
            if project_uid:
                body["query"]["bool"]["filter"] = [
                    {"term": {"project_uid": project_uid}}
                ]

            resp = self.client.search(index=self.index_name, body=body)
            results = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                src["_score"] = hit["_score"]
                results.append(src)
            return results

        try:
            return _do_search()
        except Exception:
            logger.exception("ES search failed")
            return []

    def delete_by_document_uid(self, document_uid: str) -> int:
        """删除指定文档的所有 chunk。"""
        try:
            resp = self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"document_uid": document_uid}
                    }
                },
                refresh=True,
            )
            return resp.get("deleted", 0)
        except Exception:
            logger.exception("ES delete by document_uid failed")
            return 0

    def delete_by_chunk_uids(self, chunk_uids: list[str]) -> int:
        """删除指定 chunk_uids。"""
        if not chunk_uids:
            return 0
        try:
            from elasticsearch.helpers import bulk

            actions = [
                {"_op_type": "delete", "_index": self.index_name, "_id": cuid}
                for cuid in chunk_uids
            ]
            success, errors = bulk(self.client, actions, refresh=True)
            return success
        except Exception:
            logger.exception("ES delete by chunk_uids failed")
            return 0
