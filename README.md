# Agentic RAG Studio

[![Python CI](https://github.com/YYL424/agentic-rag-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/YYL424/agentic-rag-studio/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

一个面向企业文档的 Agentic RAG 工程原型：用 LangGraph 编排文档入库、混合检索问答和增量更新，将向量检索与 Neo4j 图检索并行融合，并提供 HITL 审核、SSE、MCP 和可复现评测。

> 项目定位：可运行、可测试、适合技术作品集展示的参考实现，不宣称已经达到生产环境的高可用、权限治理和大规模性能要求。Python 是主实现；Java/Go 目录是对照原型，不与 Python 功能等价。

## 为什么值得看

- **不是单次 Prompt Demo**：入库、问答、更新是三条可持久化 LangGraph 工作流。
- **检索路径可解释**：向量检索和参数化图路径查询并发执行，返回来源、分数和检索类型。
- **更新保持一致性**：按 chunk 做文本 diff，同时清理向量与图谱 provenance，避免旧事实残留。
- **关键写入可审核**：HITL 驳回会同时阻断向量库和图数据库写入。
- **上传具备幂等语义**：SQLite 文档目录持久化文件状态，以 SHA-256 阻止重复入库，失败或驳回后可安全重试。
- **边界有防护**：上传扩展名/大小/路径限制、可选 API Key、只读 Cypher 校验、XSS 安全文本渲染。
- **结果可验证**：自动化测试、覆盖率门槛、Ruff、CI、真实 Qdrant/Neo4j roundtrip，以及带来源约束和数据哈希的检索评测。

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
| 模型兼容 | 统一结构化输出适配器；按 provider 选择 function calling / JSON mode，带 schema 约束、超时与降级 |
| GraphRAG | 问题实体驱动的邻居和最短路径查询；查询参数化，关系类型使用 allowlist |
| 检索 | Chroma、Qdrant、PGVector 统一接口；向量/图并发；可选 BGE reranker 和 Self-RAG |
| 增量更新 | 文件监听/Kafka 消费入口、快照、chunk diff、向量和图谱 provenance 联动删除 |
| Agent 编排 | LangGraph checkpointer、HITL interrupt/resume、失败更新重试 |
| 接口 | FastAPI、同步问答、SSE、上传/审核/更新、持久文档列表/删除、健康检查、内置 Web UI、MCP tools |
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

如果从项目根目录操作，Git Bash 使用 `cp python/.env.example python/.env`；PowerShell 使用
`Copy-Item python\.env.example python\.env`。`Copy-Item` 不是 Git Bash 命令。

轻量启动建议在 `.env` 中设置：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_BACKEND=local
MARKER_ENABLED=false
ENABLE_RERANKER=false
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://localhost:6333
```

`local` 模式在核心依赖中使用 Chroma 的 ONNX MiniLM 回退，首次运行会下载一次模型；Docker Compose 会将模型保存在 `model_cache` volume 中。它适合快速演示。如果接口支持 `/embeddings`，也可以设置 `EMBEDDING_BACKEND=api`；如需中文 BGE embedding、reranker、marker 和 docling：

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

也可以在项目根目录执行以下命令启动 Neo4j、Qdrant 和 API；需要 Kafka、Redis、Chroma 时增加 `--profile optional`：

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:8080/api/health/ready
```

访问：

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

上传响应中的 `file_id` 是服务端生成的受管文件标识。`GET /api/documents` 可恢复文档状态，
`DELETE /api/documents/{file_id}` 会同步清理上传文件、向量、图谱来源和目录记录。相同内容再次上传时返回
`already_exists`，不会重复写入；失败或驳回的记录会在补偿清理旧双写后允许重试。删除待审核文档还会终止对应
LangGraph checkpoint，避免旧 `thread_id` 再次恢复写入。

生产式部署应设置 `API_KEY`，并在上传、审核和更新请求中携带 `X-API-Key`。当前 Key 只保护写接口，读接口仍适合置于网关后。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/ingest/upload` | 单文件解析和入库；HITL 开启时返回待审核状态 |
| `POST` | `/api/ingest/review` | 恢复挂起流程并通过/驳回写入 |
| `POST` | `/api/ingest/batch` | 批量上传 |
| `POST` | `/api/qa/ask` | 混合检索问答 |
| `POST` | `/api/qa/ask_stream` | SSE 流式问答 |
| `GET` | `/api/documents` | 获取持久化文档目录与处理状态 |
| `DELETE` | `/api/documents/{file_id}` | 清理文件、向量、图谱 provenance 和目录记录 |
| `POST` | `/api/admin/update` | 对上传目录内文件触发增量更新 |
| `GET` | `/api/admin/stats` | 向量和图谱统计 |
| `GET` | `/api/health/live` | 进程存活检查 |
| `GET` | `/api/health/ready` | 依赖就绪检查 |

## 测试与质量门槛

```bash
cd python
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest tests -q --cov --cov-report=term-missing --cov-fail-under=70
```

测试使用 fake LLM/embedding 覆盖主工作流、HITL 拒绝语义、增量一致性、并发检索、上传安全、文档幂等、只读 Cypher、API 与评测指标。CI 在 Python 3.11、3.12、3.13 上执行同一质量门槛，并通过 service containers 对真实 Qdrant 与 Neo4j 执行写入、查询、删除 roundtrip。

服务启动并配置好模型后，可运行会自动清理测试数据的完整烟雾测试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\e2e-smoke.ps1
```

```bash
bash scripts/e2e-smoke.sh
```

脚本依次验证 readiness、上传、内容去重、目录持久化、问答和删除。如果设置了 `API_KEY`，请同时设置环境变量 `AGENTHUB_API_KEY`。

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
agentic-rag-studio/
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
├── scripts/               # 可重复的真实 API smoke test
└── docker-compose.yml
```

## 当前边界与后续方向

- Java/Go 尚未接入 Python 解析 API，也未达到 Python 主实现的功能覆盖。
- Kafka 消费和文件监听提供了组件入口，但默认 API 生命周期不会自动启动它们。
- 文档目录使用上传卷内 SQLite，适合单 API 实例；多实例需要迁移到 Postgres 等共享数据库。
- 长文档入库仍占用一次 HTTP 请求；尚未实现带取消、重试和补偿的分布式后台作业。
- CI 验证真实 Qdrant/Neo4j 存储契约，但不在公共流水线调用付费 LLM，也未覆盖 Kafka。
- API Key 不是完整的用户、租户、RBAC 与审计方案。
- 示例 golden set 规模较小；正式指标需要真实文档、人工标注和重复实验。
- 默认 memory checkpointer 适合开发；多实例部署需要配置并验证 Redis/Postgres saver。

路线图见 [project-plan.md](./docs/project-plan.md)，面试表达见 [interview-guide.md](./docs/interview-guide.md)，可直接修改的简历条目见 [resume-template.md](./docs/resume-template.md)。

## License

[MIT](./LICENSE)
