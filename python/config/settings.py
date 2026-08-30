"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # ── Embedding: "api" (OpenAI-compatible) | "local" (BGE / ONNX fallback) ──
    embedding_backend: Literal["api", "local"] = "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector Store: chroma | pgvector | qdrant ────────
    vector_store_type: Literal["chroma", "pgvector", "qdrant"] = "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    # Chroma: 留空走服务端模式 (HttpClient); 设置后走本地持久化模式 (PersistentClient, 无需 chroma server)
    chroma_path: str = ""
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"
    # Qdrant: 设置 qdrant_url 走服务端模式 (http://localhost:6333);
    # 留空则使用嵌入式本地模式 (qdrant_path), 无需 Docker
    qdrant_url: str = ""
    qdrant_path: str = "./data/qdrant"
    qdrant_api_key: str = ""

    # ── Reranker (Phase 3) ──────────────────────────────
    # 本地 BGE Reranker, 未安装 sentence-transformers 时自动降级为加权混合排序
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 20  # 重排前先粗召回的数量

    # ── Self-RAG (Phase 3) ──────────────────────────────
    enable_self_rag: bool = True
    self_rag_threshold: float = 0.6   # 检索相关性低于此分数时改写查询重检
    self_rag_max_rounds: int = 2      # 最大重检轮数

    # ── Kafka (CDC) ─────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # ── LangGraph Checkpointer (Phase 2) ────────────────
    # memory | redis | postgres
    checkpoint_backend: str = "memory"
    redis_url: str = "redis://localhost:6379"
    postgres_checkpoint_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # ── HITL 人机协同 (Phase 2) ─────────────────────────
    # 开启后知识抽取结果需人工确认才能入库
    enable_hitl: bool = False

    # ── API 安全 ─────────────────────────────────────────
    # 留空保持本地开发免鉴权；生产环境设置后，写接口必须携带 X-API-Key。
    api_key: str = ""
    max_upload_size_mb: int = Field(default=50, ge=1, le=1024)
    max_batch_files: int = Field(default=10, ge=1, le=100)
    max_question_length: int = Field(default=4000, ge=32, le=50_000)

    # ── 文档解析 (Phase 1) ──────────────────────────────
    # marker-pdf 首次运行需要下载大型模型；置 false 时走 pymupdf/pypdf 轻量路径。
    marker_enabled: bool = False
    # 语义分块: embedding 检测话题边界, 不可用时自动降级为结构分块
    semantic_chunk_enabled: bool = True
    chunk_breakpoint_threshold: int = 85
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── MCP Server (Phase 2) ────────────────────────────
    mcp_server_name: str = "agent-knowledge-hub"

    # ── LangSmith 可观测性 (Phase 5) ────────────────────
    # 设置 langsmith_api_key 后自动开启全链路追踪
    langsmith_api_key: str = ""
    langsmith_project: str = "agent-knowledge-hub"
    langsmith_tracing: bool = False

    # ── API ─────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # ── Document Store ──────────────────────────────────
    upload_dir: str = "./uploads"

    @model_validator(mode="after")
    def validate_chunk_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        if self.self_rag_max_rounds < 1:
            raise ValueError("self_rag_max_rounds 必须至少为 1")
        if self.reranker_top_n < 1:
            raise ValueError("reranker_top_n 必须至少为 1")
        return self

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
