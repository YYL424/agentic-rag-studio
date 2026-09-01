# 简历项目模板

## 推荐项目名称

**Agentic RAG Studio｜图谱增强的智能知识库（个人项目）**

技术栈：Python、LangGraph、FastAPI、Neo4j、Qdrant/Chroma、Kafka、MCP、Docker、pytest

## 无指标版本（现在即可使用）

- 设计并实现三条 LangGraph 有状态工作流，覆盖多格式文档入库、向量/知识图谱混合检索问答与 chunk 级增量更新，支持 checkpointer、HITL interrupt/resume 和失败重试。
- 将向量检索与参数化 Neo4j 邻居/最短路径查询并发执行，引入可选 cross-encoder reranker 与有限轮次 Self-RAG，并返回来源、检索类型和推理步骤。
- 为图实体和关系设计 `source::chunk_id` provenance，文档变更时联动清理向量记录和失效图事实，保留仍被其他文档引用的知识。
- 设计上传幂等与可恢复文档目录：流式计算 SHA-256、SQLite 持久化处理/HITL 状态，支持重复阻断、失败重试和文件/向量/图谱联动删除。
- 统一 OpenAI-compatible 模型的结构化输出适配，按 provider 选择 function calling/JSON mode，加入 Pydantic schema、超时、日志和降级策略。
- 完善工程安全与交付：限制上传类型/大小/路径、可选写接口 API Key、关系 allowlist、只读 Cypher、非 root Docker、健康检查及 GitHub Actions；CI 使用真实 Qdrant/Neo4j 验证存储 roundtrip。
- 建立 fake LLM/embedding 的自动化测试与 70% 覆盖率门槛，并实现带来源约束、warm-up、运行环境和 SHA-256 的 Recall@K/MRR/NDCG 评测工具。

建议从上面选择 3–4 条，不要全部堆入一段经历。

## 有真实指标后的版本

只有完成公开可复现评测后，才使用下面句式：

- 在 **[文档数/总页数]**、**[人工问题数]** 的固定评测集上，相比 **[明确 baseline]**，Recall@5 从 **[A]** 提升到 **[B]**，MRR 从 **[C]** 提升到 **[D]**；报告记录模型、配置、数据哈希与 **[重复次数]** 次运行方差。
- 在 **[硬件/并发数]** 下，将检索 P95 从 **[A ms]** 降至 **[B ms]**，主要通过向量/图查询并发、同步 SDK 线程卸载和 warm cache；使用 **[工具]** 进行 **[持续时间]** 压测。
- 对 **[修改比例]** 的文档更新，只重处理 **[chunk 数]**，写入量较全量重建下降 **[X%]**；通过故障注入验证 stale vector 与 graph provenance 均被清理。

方括号必须替换为真实结果，并保留 benchmark JSON 或 CI artifact。不要使用仓库旧文档中的示例数字。

## 面试自我介绍中的一句话

> 这个项目最核心的不是接了多少模型，而是我把 RAG 的入库审核、双存储一致性、安全查询和可复现评测做成了能被测试的工程流程。

## 技术难点 STAR 模板

**S：** 文档更新后，向量库可以按 chunk 删除，但图谱实体按名称合并，容易保留已经失效的关系。

**T：** 在不删除其他文档共享事实的前提下，让局部更新保持两种存储的一致性。

**A：** 为实体和关系增加 `source::chunk_id` provenance；快照 diff 计算 stale chunk；删除时先移除来源，再只清理 provenance 为空的事实；用重复内容、修改和删除场景做回归测试。

**R：** 系统能够区分共享事实与失效事实，修改时只重处理变化 chunk。若要量化写放大或耗时，需要补充真实规模实验。

## 简历发布前检查

- [ ] GitHub Actions 已在远端变绿。
- [ ] README 快速启动在干净环境复现成功。
- [ ] 录屏中的命令和配置不包含密钥。
- [x] 至少有一份真实 Qdrant/Neo4j roundtrip 日志；提交前仍应保存远端 CI 链接。
- [ ] 所有性能数字都有数据集、baseline、环境和原始结果。
- [ ] 明确 Python 是主实现，Java/Go 是原型。
- [ ] 能解释一个失败案例、一个安全修复和一个架构取舍。
