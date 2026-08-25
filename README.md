# AgentKnowledgeHub

[![Python CI](https://github.com/bcefghj/agent-knowledge-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/bcefghj/agent-knowledge-hub/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

一个面向企业文档的 Agentic RAG 工程原型：用 LangGraph 编排文档入库、混合检索问答和增量更新，将向量检索与 Neo4j 图检索并行融合，并提供 HITL 审核、SSE、MCP 和可复现评测。

> 项目定位：可运行、可测试、适合技术作品集展示的参考实现，不宣称已经达到生产环境的高可用、权限治理和大规模性能要求。Python 是主实现；Java/Go 目录是对照原型，不与 Python 功能等价。

## 为什么值得看

- **不是单次 Prompt Demo**：入库、问答、更新是三条可持久化 LangGraph 工作流。
- **检索路径可解释**：向量检索和参数化图路径查询并发执行，返回来源、分数和检索类型。
- **更新保持一致性**：按 chunk 做文本 diff，同时清理向量与图谱 provenance，避免旧事实残留。
- **关键写入可审核**：HITL 驳回会同时阻断向量库和图数据库写入。
- **边界有防护**：上传扩展名/大小/路径限制、可选 API Key、只读 Cypher 校验、XSS 安全文本渲染。
- **结果可验证**：自动化测试、覆盖率门槛、Ruff、CI，以及带来源约束和数据哈希的检索评测。

## 架构

```text
                         FastAPI / Web UI / MCP
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       文档入库工作流          问答工作流           增量更新工作流
    Parse → Extract → HITL    Intent / Rewrite       Snapshot / Diff
              │              ┌───────┴───────┐      Changed chunks only
              │              ▼               ▼              │
              │         Vector search    Graph search         │
              │              └───────┬───────┘              │
              │                 Rerank / Self-RAG             │
              ├──────────────────────┼───────────────────────┤
              ▼                      ▼                       ▼
     Chroma / Qdrant / PGVector    LLM answer               Neo4j
```

三条主链路：

1. `parse → extract → [review] → vector store + knowledge graph`
2. `intent/rewrite → vector retrieval || graph retrieval → rerank → self-RAG → answer`
3. `change event → snapshot diff → delete stale provenance → reprocess changed chunks`

详细设计与取舍见 [架构文档](./docs/architecture.md)。

## 已实现能力

| 模块 | 当前实现 |
|---|---|
| 文档解析 | Markdown、TXT、PDF、DOCX、PPTX、CSV、XLSX、图片；PyMuPDF/python-docx 等轻量路径，marker/docling 为可选增强 |
| 分块 | 标题层级、表格/页码元数据；相邻句向量距离的语义边界分块；不可用时结构化降级 |
| 知识抽取 | Pydantic 结构化实体、关系、事件输出；实体别名与可选 embedding 消解 |
| GraphRAG | 问题实体驱动的邻居和最短路径查询；查询参数化，关系类型使用 allowlist |
| 检索 | Chroma、Qdrant、PGVector 统一接口；向量/图并发；可选 BGE reranker 和 Self-RAG |
| 增量更新 | 文件监听/Kafka 消费入口、快照、chunk diff、向量和图谱 provenance 联动删除 |
| Agent 编排 | LangGraph checkpointer、HITL interrupt/resume、失败更新重试 |
| 接口 | FastAPI、同步问答、SSE、上传/审核/更新、健康检查、内置 Web UI、MCP tools |
| 工程治理 | 可选写接口 API Key、上传隔离、非 root 容器、Ruff、pytest + coverage、GitHub Actions |

## 快速开始

### 1. 准备环境

- Python 3.11–3.13（可选 ML 原生依赖暂不支持 Python 3.14）
- 一个 OpenAI-compatible Chat API
- Neo4j；向量库可选 Chroma、Qdrant 或 PGVector

```bash
cd python
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-core.txt
cp .env.example .env
```

核心依赖不包含本地 embedding/reranker 模型，因此轻量启动建议在 `.env` 中设置：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_BACKEND=api
MARKER_ENABLED=false
ENABLE_RERANKER=false
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://localhost:6333
```

如需本地 BGE embedding、reranker、marker 和 docling：

```bash
pip install -r requirements.txt
```

### 2. 启动依赖和 API

开发模式可只启动 Neo4j 与 Qdrant：

```bash
docker compose up -d neo4j qdrant
cd python
uvicorn api.main:app --reload --port 8080
```

也可以在项目根目录执行 `docker compose up --build` 启动 Neo4j、Qdrant 和 API；需要 Kafka、Redis、Chroma 时增加 `--profile optional`。访问：

- Web UI：`http://localhost:8080/`
- OpenAPI：`http://localhost:8080/docs`
- 就绪检查：`http://localhost:8080/api/health/ready`

### 3. 调用接口

```bash
curl -F "file=@../test/员工资料.md" http://localhost:8080/api/ingest/upload

curl -X POST http://localhost:8080/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"员工的年假规则是什么？"}'
```

生产式部署应设置 `API_KEY`，并在上传、审核和更新请求中携带 `X-API-Key`。当前 Key 只保护写接口，读接口仍适合置于网关后。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/ingest/upload` | 单文件解析和入库；HITL 开启时返回待审核状态 |
| `POST` | `/api/ingest/review` | 恢复挂起流程并通过/驳回写入 |
| `POST` | `/api/ingest/batch` | 批量上传 |
| `POST` | `/api/qa/ask` | 混合检索问答 |
| `POST` | `/api/qa/ask_stream` | SSE 流式问答 |
| `POST` | `/api/admin/update` | 对上传目录内文件触发增量更新 |
| `GET` | `/api/admin/stats` | 向量和图谱统计 |
| `GET` | `/api/health/live` | 进程存活检查 |
| `GET` | `/api/health/ready` | 依赖就绪检查 |

## 测试与质量门槛

```bash
cd python
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest tests -q --cov --cov-report=term-missing --cov-fail-under=60
```

测试使用 fake LLM/embedding 和嵌入式向量库，覆盖主工作流、HITL 拒绝语义、增量一致性、并发检索、上传安全、只读 Cypher、API 与评测指标。CI 在 Python 3.11、3.12、3.13 上执行同一质量门槛。

## 检索评测

仓库提供 5 份示例文档和 30 条人工整理问题，用于验证评测链路，不代表真实业务效果：

```bash
cd python
python benchmarks/baseline_vs_modern.py \
  --docs benchmarks/sample_docs \
  --dataset benchmarks/golden_set.jsonl \
  --top-k 5 \
  --no-reranker
```

报告包含 Recall@K、MRR、NDCG@K、P50/P95，以及配置、运行环境、数据集与文档 SHA-256。生成结果默认不提交，避免把不同机器或模型的结果混为项目结论。简历中的效果指标应替换为你在目标数据集上实际复测的数字。

## 项目结构

```text
agent-knowledge-hub/
├── python/
│   ├── agents/            # 解析、抽取、问答、更新
│   ├── orchestrator/      # 三条 LangGraph 工作流
│   ├── services/          # 文档、向量、图谱、CDC、reranker
│   ├── api/               # FastAPI + Web UI + SSE
│   ├── benchmarks/        # 数据集、指标与对比脚本
│   ├── tests/             # 单元/集成边界测试
│   └── mcp_server.py      # MCP tools
├── java/                  # Spring Boot 对照原型
├── golang/                # Gin/Go 对照原型
├── docs/                  # 架构、面试、简历与路线图
└── docker-compose.yml
```

## 当前边界与后续方向

- Java/Go 尚未接入 Python 解析 API，也未达到 Python 主实现的功能覆盖。
- Kafka 消费和文件监听提供了组件入口，但默认 API 生命周期不会自动启动它们。
- CI 目前验证核心逻辑，不启动真实 Neo4j/Kafka 的端到端环境。
- API Key 不是完整的用户、租户、RBAC 与审计方案。
- 示例 golden set 规模较小；正式指标需要真实文档、人工标注和重复实验。
- 默认 memory checkpointer 适合开发；多实例部署需要配置并验证 Redis/Postgres saver。

路线图见 [project-plan.md](./docs/project-plan.md)，面试表达见 [interview-guide.md](./docs/interview-guide.md)，可直接修改的简历条目见 [resume-template.md](./docs/resume-template.md)。

## License

[MIT](./LICENSE)
