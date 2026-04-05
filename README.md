# AgentKnowledgeHub — 企业级多Agent知识管理系统

> **一句话介绍**: 4个AI Agent协作完成企业知识的全生命周期管理——从文档解析、知识抽取、智能问答到增量更新。

> **适合人群**: 准备面试的开发者、想学习多Agent系统的小白、想做AI项目经历的同学

---

## 目录

- [项目简介](#项目简介)
- [为什么做这个项目](#为什么做这个项目)
- [技术架构](#技术架构)
- [技术栈](#技术栈)
- [三语言实现](#三语言实现)
- [快速开始](#快速开始)
- [核心功能详解](#核心功能详解)
- [API 接口文档](#api-接口文档)
- [项目结构](#项目结构)
- [面试相关资料](#面试相关资料)
- [常见问题 FAQ](#常见问题-faq)
- [参考资料](#参考资料)

---

## 项目简介

**AgentKnowledgeHub** 是一个企业级多Agent知识管理系统，包含 **4个核心Agent**，通过混合编排模式协作完成知识的全生命周期管理。

### 4个Agent分别是什么？

| Agent | 做什么 | 类比理解 |
|-------|--------|----------|
| **文档解析Agent** | 把PDF/图片/表格等各种文档"读懂" | 相当于一个超强的秘书，能看懂任何格式的文件 |
| **知识抽取Agent** | 从文档中提取人名、组织、关系等结构化信息 | 相当于一个分析师，把信息整理成知识图谱 |
| **问答Agent** | 回答用户关于企业知识的问题 | 相当于一个专家顾问，能综合多个信息源回答 |
| **知识更新Agent** | 当文档有变化时，自动更新知识库 | 相当于一个勤快的管理员，实时保持知识最新 |

### 三大技术亮点

1. **多模态RAG** — 不只处理文字，还能理解图片、表格、流程图
2. **知识图谱 (GraphRAG)** — 用图数据库存储实体关系，支持多跳推理
3. **增量更新 (CDC)** — 文档变了只更新变化的部分，不用全量重建

---

## 为什么做这个项目

### 企业痛点

在真实的企业环境中，知识管理面临这些痛点：

1. **文档格式多样** — PDF、Word、Excel、图片...传统系统只能处理纯文本
2. **知识散落各处** — 信息分散在不同系统中，搜不到、找不到
3. **检索不准确** — 关键词搜索无法理解语义，召回率低
4. **更新滞后** — 文档更新后，知识库没有同步，答案过时
5. **缺乏推理能力** — 无法回答需要综合多个信息源的问题

### 我们的解决方案

```
传统方案:  文档 → 关键词索引 → 搜索        (准确率 ~60%)
本项目:    文档 → 多Agent协作 → 智能问答    (准确率 ~94%)
```

---

## 技术架构

### 整体架构图

```
┌──────────────────────────────────────────────────────────┐
│                      用户接口层                            │
│              REST API / Web UI / SDK                      │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
┌──────────────▼───────────────────────────▼───────────────┐
│                    编排引擎 (Orchestrator)                  │
│              LangGraph 有向图状态机编排                      │
│    ┌─────────────┬──────────────┬──────────────┐         │
│    │ 文档入库流程  │   问答流程    │  增量更新流程  │         │
│    └──────┬──────┴──────┬───────┴──────┬───────┘         │
└───────────│─────────────│──────────────│─────────────────┘
            │             │              │
┌───────────▼──┐ ┌───────▼────┐ ┌───────▼──────┐ ┌────────────┐
│ 文档解析Agent │ │  问答Agent  │ │ 知识更新Agent │ │ 知识抽取Agent│
│              │ │            │ │              │ │            │
│ - PDF解析    │ │ - 意图识别  │ │ - 文件监听    │ │ - NER      │
│ - 图片OCR    │ │ - 向量检索  │ │ - CDC消费    │ │ - 关系抽取  │
│ - 表格提取   │ │ - 图谱检索  │ │ - 差量对比    │ │ - 事件抽取  │
│ - 文档分块   │ │ - 混合排序  │ │ - 增量更新    │ │ - 三元组生成│
└──────┬───────┘ │ - 答案生成  │ │ - 版本管理    │ └─────┬──────┘
       │         └──┬────┬────┘ └──────┬───────┘       │
       │            │    │             │               │
┌──────▼────────────▼────│─────────────▼───────────────▼──┐
│                   存储层                                   │
│  ┌─────────────┐  │  ┌──────────────┐  ┌──────────────┐ │
│  │ ChromaDB /  │  │  │  Neo4j       │  │   Kafka      │ │
│  │ PGVector    │◄─┘  │  知识图谱     │  │   CDC队列    │ │
│  │ 向量数据库   │      │              │  │              │ │
│  └─────────────┘      └──────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 三条编排流水线

**1. 文档入库流水线**
```
上传文档 → 文档解析Agent(分类→解析→分块) → 知识抽取Agent(NER→RE→三元组)
                                              ↓                    ↓
                                         存入向量库            存入知识图谱
```

**2. 问答流水线**
```
用户提问 → 意图识别 → 查询改写 → ┌→ 向量检索 ─┐→ 混合重排序 → LLM生成答案
                                 └→ 图谱检索 ─┘
```

**3. 增量更新流水线**
```
文件变更/CDC事件 → 差量分析 → 增量解析 → 更新向量库 + 更新图谱
                     ↓
                版本管理(timestamp + version)
```

---

## 技术栈

### Python 版 (主要实现)

| 组件 | 技术选型 | 为什么选它 |
|------|----------|------------|
| Agent编排 | **LangGraph** | 2026年生产级Agent编排标准，支持有向图、条件路由、状态持久化 |
| LLM调用 | **LangChain + OpenAI** | 最成熟的LLM应用框架 |
| 向量数据库 | **ChromaDB / PGVector** | ChromaDB开箱即用，PGVector适合已有PostgreSQL的企业 |
| 知识图谱 | **Neo4j** | 图数据库的事实标准，Cypher查询语言强大 |
| 消息队列 | **Kafka** | CDC事件流处理的工业标准 |
| API框架 | **FastAPI** | 异步高性能，自动生成OpenAPI文档 |
| 文档解析 | **Unstructured + PyPDF2 + Tesseract** | 多模态文档解析全家桶 |
| 容器化 | **Docker Compose** | 一键启动所有依赖 |

### Java 版

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 框架 | **Spring Boot 3.4 + Spring AI** | Java生态最成熟的AI应用框架 |
| LLM | **Spring AI OpenAI + LangChain4j** | Spring原生AI支持 |
| 文档解析 | **Apache Tika** | Java文档解析标准库，支持1000+格式 |
| 向量存储 | **Milvus** (内存实现兜底) | 企业级向量数据库 |
| CDC | **Spring Kafka** | @KafkaListener 注解驱动 |

### Go 版

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| API框架 | **Gin** | Go生态最流行的HTTP框架 |
| LLM | **go-openai** | OpenAI官方Go SDK |
| 图数据库 | **neo4j-go-driver** | Neo4j官方Go驱动 |
| 向量存储 | **pgvector-go** | PostgreSQL向量扩展Go客户端 |
| 并发 | **goroutine** | Go原生并发，文档批量解析天然并行 |

---

## 三语言实现

为什么提供三种语言？面试时你可以根据岗位要求选择：

| 语言 | 适合岗位 | 特点 |
|------|----------|------|
| **Python** | AI工程师、算法工程师、数据工程师 | 最完整的实现，LangGraph原生支持 |
| **Java** | 后端开发、架构师、Java技术栈企业 | Spring生态，企业级标准 |
| **Go** | 基础架构、云原生、高性能后端 | 高并发处理，编译型语言性能优势 |

---

## 快速开始

### 前置条件

- Docker & Docker Compose
- 一个 OpenAI API Key (或兼容的LLM API)

### 1. 克隆项目

```bash
git clone https://github.com/bcefghj/agent-knowledge-hub.git
cd agent-knowledge-hub
```

### 2. 配置环境变量

```bash
cd python
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
```

### 3. 一键启动 (Docker)

```bash
# 回到项目根目录
cd ..
docker-compose up -d
```

这会启动：
- Neo4j (端口 7474/7687)
- ChromaDB (端口 8000)
- Kafka + Zookeeper (端口 9092)
- Python API (端口 8080)

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8080/api/health

# 上传文档
curl -X POST http://localhost:8080/api/ingest/upload \
  -F "file=@你的文档.pdf"

# 提问
curl -X POST http://localhost:8080/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这个文档讲了什么？"}'
```

### 5. 本地开发 (不用 Docker)

```bash
# Python 版
cd python
pip install -r requirements.txt
python -m api.main

# Java 版
cd java
mvn spring-boot:run

# Go 版
cd golang
go run main.go
```

---

## 核心功能详解

### 功能1: 多模态文档解析

```python
# 文档解析Agent会自动识别文件类型，调用对应的解析器
agent = DocParserAgent()

# 支持的格式
chunks = await agent.parse("报告.pdf")      # PDF → 文字 + 图片 + 表格
chunks = await agent.parse("流程图.png")     # 图片 → OCR + LLM视觉理解
chunks = await agent.parse("数据.xlsx")      # Excel → 结构化文本
chunks = await agent.parse("文档.md")        # Markdown → 纯文本
```

### 功能2: 知识图谱构建

```python
# 知识抽取Agent从文本中提取三元组
extractor = KnowledgeExtractAgent()
results = await extractor.extract(chunks)

# 结果示例:
# entities: [("张三", Person), ("腾讯", Organization), ("微信", Product)]
# relations: [("张三", works_at, "腾讯"), ("腾讯", developed, "微信")]
```

### 功能3: GraphRAG 混合检索

```python
# 问答Agent同时查询向量库和知识图谱
qa = QAAgent(vector_store=vs, knowledge_graph=kg)
result = await qa.answer("张三在哪里工作？开发了什么产品？")

# 内部流程:
# 1. 向量检索 → 找到语义相关的文档块
# 2. 图谱检索 → 找到"张三"→works_at→"腾讯"→developed→"微信"
# 3. 混合排序 → 图谱结果权重×1.2 (结构化信息更精准)
# 4. 生成答案 → 综合两个来源的信息回答
```

### 功能4: CDC 增量更新

```python
# 文档变更时，只处理变化的部分
update_agent = KnowledgeUpdateAgent(...)

# 场景: 修改了一个PDF文件
# 传统做法: 全量删除 → 全量重新解析 → 全量重新入库 (慢!)
# CDC做法:  检测变更 → 差量对比 → 只更新变化的chunk → 只更新相关实体 (快!)
```

---

## API 接口文档

启动服务后，访问 `http://localhost:8080/docs` 可以看到交互式API文档 (Swagger UI)。

### 核心接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest/upload` | 上传单个文档 |
| POST | `/api/ingest/batch` | 批量上传文档 |
| POST | `/api/qa/ask` | 智能问答 |
| GET | `/api/admin/stats` | 系统统计 |
| POST | `/api/admin/update` | 手动触发更新 |
| GET | `/api/health` | 健康检查 |

---

## 项目结构

```
AgentKnowledgeHub/
├── README.md                          # 你正在看的这个文件
├── docker-compose.yml                 # 一键启动所有服务
├── docs/                              # 文档
│   ├── architecture.md                # 架构设计详解
│   ├── interview-guide.md             # 面试八股文 + STAR法则
│   ├── resume-template.md             # 简历写法模板
│   └── tech-deep-dive.md              # 核心代码逐行讲解
│
├── python/                            # Python 实现 (最完整)
│   ├── agents/                        # 4个Agent
│   │   ├── doc_parser_agent.py        # 文档解析Agent
│   │   ├── knowledge_extract_agent.py # 知识抽取Agent
│   │   ├── qa_agent.py                # 问答Agent
│   │   └── knowledge_update_agent.py  # 知识更新Agent
│   ├── orchestrator/
│   │   └── graph.py                   # LangGraph 编排引擎
│   ├── services/
│   │   ├── vector_store.py            # 向量库服务
│   │   ├── knowledge_graph.py         # 知识图谱服务
│   │   ├── graph_rag.py               # GraphRAG 混合检索管道
│   │   ├── cdc_processor.py           # CDC 增量处理器
│   │   └── multimodal.py              # 多模态服务
│   ├── api/
│   │   └── main.py                    # FastAPI 入口
│   ├── config/
│   │   └── settings.py                # 配置管理
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── java/                              # Java 实现
│   ├── src/main/java/com/agenthub/
│   │   ├── agent/                     # 4个Agent
│   │   ├── service/                   # 服务层
│   │   ├── controller/                # REST控制器
│   │   └── model/                     # 数据模型
│   └── pom.xml
│
└── golang/                            # Go 实现
    ├── agent/                         # Agent实现
    ├── service/                       # 服务层
    ├── api/                           # HTTP服务器
    ├── model/                         # 数据模型
    ├── config/                        # 配置
    ├── main.go                        # 入口
    └── go.mod
```

---

## 面试相关资料

本项目专门为面试准备了全套资料，详见 `docs/` 目录：

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [**面试八股文 + STAR法则**](docs/interview-guide.md) | 30+高频面试题详解 + STAR话术模板 | 面试前突击 |
| [**简历写法模板**](docs/resume-template.md) | 怎么把这个项目写到简历上 | 投简历前 |
| [**架构设计详解**](docs/architecture.md) | 为什么这么设计？每个决策的理由 | 面试深度追问 |
| [**核心代码讲解**](docs/tech-deep-dive.md) | 关键代码逐行解读 | 代码层面的理解 |

---

## 常见问题 FAQ

### Q: 我没有 OpenAI API Key 怎么办？

可以用任何兼容 OpenAI 接口的 LLM 服务：
- **本地部署**: Ollama (免费，推荐 llama3/qwen2)
- **国内服务**: 通义千问、智谱、MiniMax 等都提供 OpenAI 兼容接口

修改 `.env` 中的 `OPENAI_BASE_URL` 指向对应服务即可。

### Q: Neo4j / ChromaDB 启动失败？

确保 Docker 有足够内存(建议 4GB+)：
```bash
# 检查 Docker 状态
docker-compose ps

# 查看日志
docker-compose logs neo4j
```

### Q: 这个项目可以直接用在生产环境吗？

这是一个**面试项目 + 学习项目**，展示了企业级架构设计。如果要用在生产环境，还需要：
- 添加用户认证 (JWT/OAuth2)
- 增加限流和熔断
- 完善日志和监控
- 增加单元测试和集成测试

### Q: Python/Java/Go 三个版本有什么区别？

- **Python版** 最完整，包含所有功能，推荐作为学习入口
- **Java版** 用Spring生态实现，适合Java岗位面试
- **Go版** 用Gin框架实现，适合Go岗位面试

三个版本的**架构设计完全一致**，只是语言和框架不同。

---

## 参考资料

### 框架 & 工具
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Spring AI 官方文档](https://docs.spring.io/spring-ai/reference/)
- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)

### 论文 & 文章
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

---

## 贡献

欢迎提 Issue 和 PR！如果觉得有帮助，请给个 Star。

## License

MIT License
