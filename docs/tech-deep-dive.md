# 技术深挖：从 RAG Demo 到可验证工程

## 1. HITL 的语义必须覆盖所有副作用

入库图在抽取后分成向量和图谱两个写分支。仅在驳回时清空 `extractions` 不够，因为原始 `chunks` 仍存在，向量分支仍会写入。

当前实现把审核结果写入显式状态 `ingest_approved`，两个 store 节点都做门禁。这个问题的通用经验是：**审批控制的是副作用，而不是某个中间变量**。回归测试同时断言 `vectors_stored == 0` 和 `entities_stored == 0`。

## 2. 图谱增量更新需要 provenance

实体按名称 `MERGE` 后，简单的 `DELETE WHERE source=...` 会遇到两个问题：节点可能没有 source，或者同一事实由多个文档支持。项目为节点和关系维护 provenance 数组：

```text
uploads/a.md::docA#chunk-3
uploads/b.md::docB#chunk-7
```

删除文档或 chunk 时，先从数组移除匹配来源；只有数组为空时才删除事实。这个方案提供了基础可追溯性，但不是完整 temporal knowledge graph：版本、有效时间、冲突事实和置信度融合仍需独立建模。

## 3. LLM 不应拥有图数据库写权限

早期 GraphRAG 常见做法是让 LLM 生成 Cypher 后直接执行。这既有写入风险，也可能产生笛卡尔积等高成本查询。

当前主问答链路只做两类模板查询：

- `get_neighbors(entity, hops)`
- `find_paths(entity_a, entity_b, max_hops)`

实体名作为参数传入，跳数被限制。关系类型虽然必须插入 Cypher 语法位置，但先经过固定 allowlist。MCP 的自由查询入口只允许单条只读查询；真实部署仍应使用数据库只读凭据和事务超时形成纵深防御。

## 4. Async API 不等于所有调用都是异步

Chroma、Qdrant 和部分解析 SDK 提供同步接口。在 `async def` 中直接调用会阻塞事件循环，使并发请求串行化。项目将这些调用放入 `asyncio.to_thread`，并使用 `asyncio.gather` 并发：

```text
question ─┬─ vector retrieval ─┐
          └─ graph retrieval  ─┴─ merge
```

这只是正确性层面的优化；线程池大小、数据库连接池和 P95 仍需要负载测试决定。

## 5. 语义分块应可降级、可测试

实现先按中英文句末标记切句，再计算相邻句向量余弦距离：

```text
distance(i) = 1 - cosine(sentence_i, sentence_i+1)
```

超过距离百分位阈值的位置成为候选边界，同时受 chunk 最大长度约束。embedding 通过接口注入，因此测试使用确定性二维 fake vector，无需下载模型。没有 embedding 时走结构分块，使轻量安装仍可运行。

## 6. 检索指标必须绑定来源

只判断“返回内容包含 golden fragment”会在多个文档重复同一句时产生假命中。评测器同时检查：

```text
normalized content matches golden
AND basename(result.source) == basename(sample.source_doc)
```

评测排除 warm-up 查询，输出 Recall@K、MRR、NDCG@K、P50/P95，并保存配置、平台、Python 版本和输入 SHA-256。示例集规模小，因此报告只能用于回归和演示，不能直接当作业务效果。

## 7. 为什么区分 liveness 与 readiness

- `/api/health/live` 只证明 Web 进程能响应，适合容器重启判断。
- `/api/health/ready` 验证向量库和 Neo4j 初始化状态，依赖失败时返回 503，避免把流量发给不可服务实例。
- `/api/health` 返回 degraded 和依赖明细，适合人工诊断。

如果把外部依赖故障等同于进程死亡，编排器会反复重启本身健康的 API，放大故障。

## 8. 下一层工程问题

真正生产化时，最值得继续深挖的是：

- 用 job/outbox/补偿重放解决跨库最终一致性。
- 用租户作用域实体 ID 代替全局按名称合并。
- 为检索分数做离线调权与 confidence calibration。
- 对 Kafka offset、重复事件和死信队列做故障注入。
- 在真实语料上建立版本化评测集和持续回归。
