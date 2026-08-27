from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Mini Agent"
    debug: bool = True
    mock_llm: bool = True

    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4.1-mini"

    # ── 多模型角色配置（多Agent架构） ──
    # 每个角色可独立指定模型，空则回退到 openai_model
    classifier_model: str = ""        # 三分类+四分类，轻量即可
    evaluator_model: str = ""         # 质量评判 completeness/groundedness
    generator_model: str = ""         # 最终回答生成（核心输出，建议最强模型）
    doc_generator_model: str = ""     # 文档格式化生成（PPT/Word/Excel/图）

    # mysql
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "mini_agent"

    # Reids
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    chat_context_limit: int = 50
    chat_recent_conversation_limit: int = 20
    chat_cache_ttl_seconds: int = 7 * 24 * 3600
    summary_trigger_message_count: int = 40

    # kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_cache_invalidation_topic: str = "cache-invalidation-topic"
    kafka_consumer_group: str = "mini-agent-cache-invalidation-group"

    # 文件上传配置
    upload_chunk_size_bytes: int = 5 * 1024 * 1024
    upload_max_file_size_bytes: int = 200 * 1024 * 1024
    upload_session_expire_hours: int = 24

    # 后续合并文档 complete_upload 会用到
    kafka_document_parse_topic: str = "document-parse-topic"

    # MinIO 配置
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_documents: str = "mini-agent-documents"

    # Elasticsearch 配置
    es_host: str = "http://127.0.0.1:9200"
    es_index_chunks: str = "mini-agent-chunks"

    # Embedding 配置（独立于 LLM，可对接不同 API）
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    embedding_batch_size: int = 10   # DashScope text-embedding-v4 单批上限 10
    embedding_worker_batch_size: int = 50  # 每轮取多少条待处理 chunk
    parse_worker_batch_size: int = 5   # 每轮取多少条待解析文档
    faiss_index_path: str = str(BASE_DIR / "data" / "faiss")

    # 文档解析与分块配置
    parse_parent_block_size_bytes: int = 1 * 1024 * 1024
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 200
    chunk_max_chars: int = 1800

    # HanLP 分句（网络不通时设为 false，使用正则兜底）
    hanlp_enabled: bool = True

    # Reranker 精排（需要 sentence-transformers + BAAI/bge-reranker-v2-m3）
    reranker_enabled: bool = True
    reranker_model: str = str(BASE_DIR / "data" / "models" / "BAAI" / "bge-reranker-v2-m3")
    # HuggingFace 镜像（国内网络可用 https://hf-mirror.com）
    hf_endpoint: str = "https://hf-mirror.com"

    # Tika Server 配置
    tika_server_url: str = "http://127.0.0.1:9998"
    tika_timeout_seconds: int = 300
    tika_stream_chunk_size_bytes: int = 8192

    # 文档解析 Kafka Topic
    kafka_document_parse_topic: str = "document-parse-topic"

    # Auth
    jwt_secret_key: str = ""
    jwt_expire_hours: int = 72
    initial_admin_username: str = "admin"
    initial_admin_password: str = ""

    # System prompt（admin 可动态修改，持久化到 data/system_prompt.json）
    system_prompt: str = ""

    # Guest login
    guest_question_limit: int = 3
    guest_document_limit: int = 2
    registration_enabled: bool = False
    contact_admin_email: str = ""

    # MCP 联网搜索
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"  # duckduckgo | brave | bing
    brave_search_api_key: str = ""
    bing_search_api_key: str = ""
    web_search_max_results: int = 5

    # Neo4j 图数据库（知识图谱）
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # 场景知识库
    scenario_kb_enabled: bool = True
    scenario_match_threshold: float = 0.58
    scenario_kb_max_results: int = 5
    scenario_faiss_index_path: str = str(BASE_DIR / "data" / "scenario_faiss")

    # 文档生成输出目录
    document_output_dir: str = str(BASE_DIR / "data" / "output")

    templates_dir: str = str(BASE_DIR / "templates")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()