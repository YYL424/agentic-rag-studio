# AgentKnowledgeHub 架构说明

## 1. 定位

AgentKnowledgeHub 是一个 Agentic RAG 工程原型。它把文档解析、结构化知识抽取、混合检索问答和增量维护拆成职责明确的组件，再由 LangGraph 编排状态、分支、暂停和恢复。

这里的“Agent”指带 LLM 决策或结构化输出的任务角色，不代表多个自治 Agent 通过消息协商。当前系统更准确的表述是：**四个领域 Agent + 三条有状态工作流**。

## 2. 运行时组件

| 层 | 组件 | 职责 |
|---|---|---|
| 接口 | FastAPI / Web UI / MCP | 上传、文档管理、问答、审核、更新、统计、健康检查和工具互操作 |
| 编排 | LangGraph | 入库、问答、更新工作流；checkpointer；HITL interrupt/resume |
| Agent | Parser / Extract / QA / Update | 文档理解、结构化抽取、检索生成和一致性更新 |
| 检索 | VectorStoreService / KnowledgeGraphService | Chroma/Qdrant/PGVector 语义检索和 Neo4j 图查询 |
| 元数据 | DocumentRegistry (SQLite) | 文件状态、内容哈希、处理计数、HITL thread 与删除目录 |
| 可选组件 | Reranker / Kafka / Watchdog / LangSmith | 精排、事件消费、文件监听和链路追踪 |

## 3. 三条核心链路

### 3.1 文档入库

```text
file
  → UUID managed filename + streaming SHA-256
  → idempotency registry (processing / review / success / failed)
  → classify / parse
  → structure-aware chunk
  → entity + relation + event extraction
  → optional HITL review
  ├→ vector upsert
  └→ graph upsert with provenance
```

关键语义：

- `DocumentChunk` 保存 `doc_id`、`chunk_id`、来源、页码、标题路径和内容类型。
- LLM 抽取使用 Pydantic schema；统一适配器按 provider 选择 function calling 或 JSON mode，并设置超时与回退。
- HITL 开启时，工作流在 `interrupt()` 处挂起；驳回后向量与图谱写入都返回 0。
- 同内容重复上传返回已有 `file_id`；失败或驳回的记录重试前先按旧 `file_id` 补偿清理两个存储。
- 删除待审核文档时先以 rejected 恢复挂起图，再删除数据和 checkpoint；旧 thread ID 不能再次恢复写入。
- 图谱实体和关系保存 `source::chunk_id` provenance，支持后续精确清理。
- 两个存储分支并行，但当前没有跨数据库事务；生产环境需要 outbox、补偿任务或幂等作业状态。

### 3.2 混合检索问答

```text
question
  → intent classification + query rewrite
  ├→ vector searches for rewritten queries
  └→ parameterized graph neighbor/path searches
  → deduplicate + weighted merge
  → optional cross-encoder rerank
  → relevance check / bounded Self-RAG retry
  → answer + sources + confidence
```

向量与图检索通过 `asyncio.gather` 并发执行。图检索不再让 LLM 生成任意 Cypher，而是把抽取到的实体名传给参数化的邻居和最短路径查询。MCP 暴露的 Cypher 工具也经过单语句、只读关键字和允许起始子句校验。

当前混排权重是启发式规则，置信度也不是统计校准概率。若要进入真实业务，需要在标注集上调权并做 calibration。

### 3.3 增量更新

```text
created / modified / deleted
  → parse current document
  → compare with previous snapshot
  → identify stale and changed chunk IDs
  → remove vector records and graph provenance
  → extract/store changed chunks only
  → save new snapshot
```

快照记录 chunk ID 和内容。修改时以 `(chunk_id, content)` 判定不变项，防止内容重复导致错误去重；旧 chunk 会同时从向量库删除并从图事实的 provenance 中移除。实体或关系仍有其他来源时保留，没有来源时清理。

Kafka consumer 和文件 watcher 已提供组件入口，但当前 FastAPI 生命周期不会自动启动它们；默认可通过管理 API 显式触发更新。

## 4. 文档解析与分块

解析器按格式选择轻量实现：

- PDF：可选 marker，随后 PyMuPDF，最后 pypdf。
- DOCX/PPTX：可选 docling，随后 python-docx/python-pptx。
- 图片：OCR，必要时尝试多模态 LLM 描述。
- Markdown/表格/文本：保留标题、表格、页码等结构元数据。

语义分块计算相邻句 embedding 的余弦距离，以配置的百分位阈值寻找主题边界，再施加 `chunk_size` 硬上限。没有 embedding 或调用失败时退化为结构分块。同步解析、embedding 后端和本地向量库操作被放入 worker thread，避免阻塞 FastAPI 事件循环。

## 5. 数据与安全边界

- 上传文件名只用于展示，磁盘名使用 UUID；文件必须属于允许扩展名并受大小限制。
- 上传流同步计算 SHA-256；目录记录与文件放在同一持久卷，Web UI 刷新后可恢复状态。
- 管理更新只能访问 `upload_dir` 内的解析后绝对路径。
- 配置 `API_KEY` 后，上传、审核和更新接口要求 `X-API-Key`。
- LLM 关系类型必须进入固定 allowlist，其他值降级为 `RELATED_TO`。
- Web UI 对文件名、问题、回答和元信息使用文本节点或 HTML 转义。
- 容器使用非 root 用户，并区分 liveness 与 dependency readiness。

这些措施不等价于完整生产安全。当前仍缺少用户身份、RBAC、租户隔离、配额、审计日志、恶意文件扫描和查询成本控制。

## 6. 关键设计取舍

### 为什么同时使用向量库和图数据库？

向量检索适合语义近似和原文证据，图查询适合显式关系和路径。两者并行可以降低串行延迟，但也引入双写一致性和排序校准问题。本项目用 provenance 与幂等 upsert 解决基本删除一致性，尚未实现跨库事务。

### 为什么不接受 LLM 生成的任意 Cypher？

即使做关键字过滤，生成式查询仍可能产生高成本或越权语句。主问答链路因此只调用预定义模板，并把用户内容作为参数。只读 Cypher 入口保留给受控 MCP 场景，但部署时仍应叠加数据库只读账号和超时限制。

### 为什么重型 ML 是可选依赖？

marker、docling、sentence-transformers 会显著增加镜像、下载和原生依赖复杂度。核心依赖可以使用 API embedding 和轻量解析器启动；完整依赖用于离线或资源充足环境。

### 为什么当前文档目录使用 SQLite？

作品集和本地单实例部署需要的是可恢复状态、内容幂等和低运维成本。SQLite 与上传文件放在同一 volume，能够原子维护单条目录记录，且不增加新服务。它不是多实例作业系统：横向扩容时应把目录和 job 状态迁移到 Postgres，并增加队列、租约、心跳、取消和补偿重放。

## 7. 验证边界

- 单元测试使用 fake LLM/embedding，覆盖工作流分支和失败语义。
- GitHub Actions 在 Python 3.11–3.13 执行 Ruff 和 70% 覆盖率门槛。
- 独立 CI job 启动真实 Qdrant 与 Neo4j，验证写入、查询和按来源清理；测试数据使用随机 ID 并在 `finally` 中删除。
- `scripts/e2e-smoke.ps1` / `.sh` 在本地验证 readiness、真实模型入库、去重、问答和删除，但公共 CI 不使用个人模型密钥。

## 8. 已知限制

- 文档目录是单实例 SQLite，不支持多个 API 副本并发调度长任务。
- 入库仍是请求内工作流，没有持久 job queue、取消、租约和跨库补偿。
- CI 有真实 Neo4j/Qdrant 存储 roundtrip，但没有真实 LLM/Kafka 端到端测试。
- 默认 memory checkpointer 不支持多实例恢复。
- 图实体以名称合并，复杂实体身份需要 tenant/domain scoped ID。
- 评测集只有 5 份示例文档和 30 个问题，仅验证评测管线。
- Java/Go 是对照原型，功能和测试覆盖不等同于 Python 主实现。

下一步见 [project-plan.md](./project-plan.md)。
