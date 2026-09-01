# Agentic RAG Studio 面试指南

## 90 秒项目介绍

> 我做了一个面向企业文档的 Agentic RAG 知识库。Python 主实现用 LangGraph 编排三条工作流：文档解析和知识入库、向量与图谱并行检索问答、以及基于 chunk diff 的增量更新。系统支持 Chroma/Qdrant/PGVector 和 Neo4j，知识抽取通过 provider-aware 适配器获得结构化输出，关键写入可通过 HITL 审核。工程上我重点解决了双存储一致性、上传幂等和安全边界：每条图事实记录来源 chunk，相同文件通过内容哈希去重，文档变化或删除时同时清理旧向量和旧 provenance；主问答链路只使用参数化图查询。项目有覆盖率门槛、CI、真实 Qdrant/Neo4j roundtrip 和带数据哈希的检索评测。当前它仍是工程原型，多实例作业、租户权限和业务规模评测是下一阶段。

## 面试官最可能追问的 12 个问题

### 1. 这是真正的多 Agent 吗？

严格说，它是四个领域 Agent 组成的 Agentic workflow，不是自治 Agent 之间自由协商。LangGraph 管理状态、分支、HITL 和重试；QA 内部又并发调用向量和图检索。用这个表述比泛称“多 Agent 平台”更准确。

### 2. 为什么需要 GraphRAG？

向量库擅长找语义相近原文，但对“某人属于哪个团队、两个实体之间是什么路径”这类显式关系问题不稳定。图数据库补充实体邻居和最短路径，向量原文则提供可引用证据。两者是互补而不是替代。

### 3. 如何防止 LLM 生成危险 Cypher？

主问答链路不执行 LLM 自由生成 Cypher，只调用参数化的邻居/路径模板。关系类型经过 allowlist。MCP 的查询入口还有只读、单语句校验；更严格的部署应使用 Neo4j 只读账号、事务超时和查询预算。

### 4. 文档更新后如何避免旧知识残留？

每个图实体和关系保存 `source::chunk_id` provenance。修改时比较新旧快照，先按 stale chunk ID 删除向量和对应 provenance，再只处理新增/变化 chunk。某事实有多个来源时，只移除当前来源；没有来源才删除。

### 5. 双写失败怎么办？

当前写入是幂等的，但没有跨 Qdrant/Neo4j 事务。如果一个分支成功、另一个失败，会出现暂时不一致。这是我明确保留的生产化缺口：下一步会引入 ingestion job 状态、outbox 和补偿重放，并以 provenance 做幂等键。

### 6. HITL 驳回为什么是一个值得测试的点？

工作流从审核节点分叉到两个存储。如果只清空抽取结果，向量分支仍可能写入原文。修复方式是在状态中保存显式 `ingest_approved`，两个写节点都检查它，并用回归测试验证驳回后写入计数均为 0。

### 7. 如何避免异步 API 被同步 SDK 阻塞？

Chroma/Qdrant 客户端以及部分解析器是同步的。服务层使用 `asyncio.to_thread` 包装这些调用，向量和图分支使用 `asyncio.gather`。真实性能结论仍需在固定并发和数据规模下压测。

### 8. 语义分块是怎么做的？

先按句子切分，计算相邻句 embedding 的余弦距离，以百分位阈值确定主题断点，再受 chunk 大小约束。模型不可用时回退到标题/段落结构分块。该实现避免依赖已归档的实验包，也便于注入 fake embedding 测试。

### 9. 评测可信吗？

评测计算 Recall@K、MRR、NDCG 和延迟，并要求命中的内容和来源文档同时匹配；还排除 warm-up，记录配置、环境与 SHA-256。仓库样例只有 30 题，因此只能证明评测工具可运行，不能代表业务准确率。简历数字必须来自更大的人工标注集和重复实验。

### 10. 为什么限制 Python 3.11–3.13？

核心 Python 代码本身可以在更新版本解释器上运行，但 marker、sentence-transformers、pyarrow 等可选依赖包含原生扩展。在 Python 3.14 环境观察到兼容性异常，所以项目主动声明经过目标依赖生态支持的版本范围，而不是掩盖风险。

### 11. 重复上传和失败重试怎么处理？

API 在流式保存文件时计算 SHA-256，并在上传 volume 内的 SQLite 目录用唯一索引登记内容哈希。并发或重复上传返回已有 `file_id`，不会再次抽取和双写；状态为 failed/rejected 或文件被手动移除时允许重试。这个方案适合单实例演示，多实例应迁移到共享数据库并使用 job lease。

### 12. 为什么结构化输出还需要一层适配器？

OpenAI-compatible 并不代表所有 provider 都完整支持 function calling。适配器集中注入 Pydantic JSON schema、设置超时并记录失败；默认 OpenAI 类 provider 先走 function calling 再降级 JSON mode，DeepSeek 类接口直接走 JSON mode。Agent 不再各自维护一套脆弱的解析逻辑。

## 可以展示的代码路径

- 工作流和 HITL：`python/orchestrator/graph.py`
- 并发混合检索：`python/agents/qa_agent.py`
- 增量一致性：`python/agents/knowledge_update_agent.py`
- provenance 与安全图查询：`python/services/knowledge_graph.py`
- 结构化模型适配：`python/services/structured_llm.py`
- 文档幂等与状态：`python/services/document_registry.py`
- 上传和健康检查：`python/api/main.py`
- 评测可追溯性：`python/benchmarks/`
- 回归测试：`python/tests/`

## 不要这样回答

- 不要说“准确率提升到 94%”，除非能给出数据集、基线、运行配置和原始报告。
- 不要说“生产级”，应说“按生产问题设计的工程原型”。
- 不要说 Java/Go 与 Python 功能一致，它们目前是对照原型。
- 不要把启发式 confidence 解释成统计概率。
- 不要说 CDC 已自动运行；当前默认 API 没有自动启动 watcher/Kafka consumer。

## 建议现场演示顺序

1. 打开 `/docs` 和 `/api/health/ready`。
2. 上传示例文档，展示 chunk 和结构化抽取。
3. 刷新页面展示文档仍存在，再重复上传一次展示内容去重。
4. 开启 HITL，在网页中演示一次驳回不写入、一次通过。
5. 提问并展示向量/图来源和 SSE。
6. 删除文档并展示向量/图谱计数回落。
7. 最后展示真实存储 CI 和 benchmark 报告，而不是只展示聊天页面。
