"""
LangGraph 编排引擎 — 4 Agent 混合编排 (LangGraph 1.x 现代范式)

编排模式:
  1. 文档入库流程: DocParser → KnowledgeExtract → [HITL Review] → (VectorStore + KnowledgeGraph)
  2. 问答流程: Query → QA Agent → (VectorRetrieval ∥ GraphRetrieval) → Answer
  3. 增量更新流程: CDC Event → UpdateAgent → (Diff → Parse → Store)

现代化特性 (LangGraph 1.x):
  - Checkpointer 状态持久化: 长流程断点续跑, 分布式多实例部署
  - interrupt() 人机协同 (HITL): 知识抽取结果经人工审核后入库
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from agents.knowledge_update_agent import (
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
    UpdateResult,
)
from agents.qa_agent import QAAgent, QAResult
from config import settings
from services.entity_resolver import EntityResolver
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    UPDATE = "update"


# ── State Schemas (TypedDict for langgraph >= 0.5) ──────────

class IngestState(TypedDict, total=False):
    """文档入库流程状态"""
    file_paths: list[str]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: int
    entities_stored: int
    review_required: bool
    review_note: str
    ingest_approved: bool
    messages: Annotated[list, add_messages]


class QAState(TypedDict, total=False):
    """问答流程状态"""
    question: str
    result: QAResult | None
    messages: Annotated[list, add_messages]


class UpdateState(TypedDict, total=False):
    """增量更新流程状态"""
    changes: list[DocumentChange]
    results: list[UpdateResult]
    messages: Annotated[list, add_messages]


# ── Checkpointer Factory ────────────────────────────────────

def create_checkpointer(backend: str | None = None):
    """
    创建 Checkpointer: memory (默认) | redis | postgres
    redis/postgres 未安装或不可用时自动降级为内存版
    """
    backend = backend or settings.checkpoint_backend

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver
            logger.info("using RedisSaver checkpointer: %s", settings.redis_url)
            return RedisSaver.from_conn_string(settings.redis_url)
        except Exception as e:
            logger.warning("RedisSaver unavailable (%s), falling back to InMemorySaver", e)

    elif backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("using PostgresSaver checkpointer: %s", settings.postgres_checkpoint_dsn)
            saver = PostgresSaver.from_conn_string(settings.postgres_checkpoint_dsn)
            saver.setup()
            return saver
        except Exception as e:
            logger.warning("PostgresSaver unavailable (%s), falling back to InMemorySaver", e)

    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:  # 兼容旧版 langgraph-checkpoint
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    logger.info("using InMemorySaver checkpointer")
    return InMemorySaver()


# ── Workflow Builder ─────────────────────────────────────────

def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
    checkpointer: Any = None,
    enable_hitl: bool | None = None,
) -> dict[str, Any]:
    """
    构建三条编排流水线，返回 {"ingest": graph, "qa": graph, "update": graph}

    所有 graph 均编译 checkpointer, 支持断点续跑
    """
    enable_hitl = settings.enable_hitl if enable_hitl is None else enable_hitl
    # HITL 依赖 checkpointer (interrupt 需要持久化挂起状态)
    checkpointer = checkpointer or create_checkpointer()

    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent(entity_resolver=EntityResolver())
    qa_agent = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    return {
        "ingest": _build_ingest_graph(
            doc_parser, extractor, vector_store, knowledge_graph,
            checkpointer=checkpointer, enable_hitl=enable_hitl,
        ),
        "qa": _build_qa_graph(qa_agent, checkpointer=checkpointer),
        "update": _build_update_graph(update_agent, checkpointer=checkpointer),
    }


# ── Ingest Pipeline ─────────────────────────────────────────

def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
    checkpointer: Any,
    enable_hitl: bool,
) -> Any:

    async def parse_documents(state: IngestState) -> dict:
        file_paths = state.get("file_paths", [])
        chunks = await doc_parser.parse_batch(file_paths)
        return {"chunks": chunks}

    async def extract_knowledge(state: IngestState) -> dict:
        chunks = state.get("chunks", [])
        extractions = await extractor.extract(chunks)
        return {"extractions": extractions}

    async def review_extractions(state: IngestState) -> dict:
        """HITL 人机协同节点: 知识入库前人工审核"""
        extractions = state.get("extractions", [])
        entities = [
            {"name": e.name, "type": e.type, "description": e.description}
            for ext in extractions for e in ext.entities
        ]
        relations = [
            {"head": r.head, "relation": r.relation, "tail": r.tail, "confidence": r.confidence}
            for ext in extractions for r in ext.relations
        ]

        # interrupt: 流程在此挂起, 等待人工确认 (通过 Command(resume=...) 恢复)
        reviewed = interrupt({
            "type": "hitl_review",
            "entities": entities[:100],
            "relations": relations[:100],
            "question": "请审核以上知识抽取结果，确认或修正后提交",
        })

        # 审核通过 (resume=None) → 原样入库
        if reviewed is None:
            return {"review_note": "approved", "ingest_approved": True}

        # 审核驳回 (resume={"approved": False}) → 不入库
        if isinstance(reviewed, dict) and reviewed.get("approved") is False:
            return {"extractions": [], "review_note": "rejected", "ingest_approved": False}

        # 审核修正 (resume 携带修正后的抽取结果) → 使用修正版
        if isinstance(reviewed, dict) and "extractions" in reviewed:
            return {
                "extractions": reviewed["extractions"],
                "review_note": "corrected",
                "ingest_approved": True,
            }

        return {"review_note": "approved", "ingest_approved": True}

    async def store_vectors(state: IngestState) -> dict:
        if state.get("ingest_approved") is False:
            return {"vectors_stored": 0}
        chunks = state.get("chunks", [])
        count = 0
        if vector_store and chunks:
            count = await vector_store.add_chunks(chunks)
        return {"vectors_stored": count}

    async def store_graph(state: IngestState) -> dict:
        if state.get("ingest_approved") is False:
            return {"entities_stored": 0}
        extractions = state.get("extractions", [])
        chunks = state.get("chunks", [])
        chunk_sources = {c.chunk_id: c.metadata.get("source", "") for c in chunks}
        entity_count = 0
        if knowledge_graph:
            for ext in extractions:
                source = chunk_sources.get(ext.source_chunk_id, "")
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(
                        ent,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
                    entity_count += 1
                for rel in ext.relations:
                    await knowledge_graph.add_relation(
                        rel,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
        return {"entities_stored": entity_count}

    graph = StateGraph(IngestState)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")

    if enable_hitl:
        # HITL 模式: extract → review → store (审核不过则不入库)
        graph.add_node("review", review_extractions)
        graph.add_edge("extract", "review")
        graph.add_edge("review", "store_vectors")
        graph.add_edge("review", "store_graph")
    else:
        graph.add_edge("extract", "store_vectors")
        graph.add_edge("extract", "store_graph")

    graph.add_edge("store_vectors", END)
    graph.add_edge("store_graph", END)

    return graph.compile(checkpointer=checkpointer)


# ── QA Pipeline ──────────────────────────────────────────────

def _build_qa_graph(qa_agent: QAAgent, checkpointer: Any) -> Any:

    async def process_question(state: QAState) -> dict:
        question = state.get("question", "")
        result = await qa_agent.answer(question)
        return {"result": result}

    graph = StateGraph(QAState)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=checkpointer)


# ── Update Pipeline ──────────────────────────────────────────

def _build_update_graph(update_agent: KnowledgeUpdateAgent, checkpointer: Any) -> Any:

    async def process_updates(state: UpdateState) -> dict:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {"results": results}

    def should_continue(state: UpdateState) -> str:
        results = state.get("results", [])
        failed = [r for r in results if not r.success]
        if failed:
            return "retry"
        return "done"

    async def retry_failed(state: UpdateState) -> dict:
        results = state.get("results", [])
        failed_changes = [r.change for r in results if not r.success]
        retried = await update_agent.process_batch(failed_changes)
        all_results = [r for r in results if r.success] + retried
        return {"results": all_results}

    graph = StateGraph(UpdateState)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)

    graph.set_entry_point("process")
    graph.add_conditional_edges("process", should_continue, {"retry": "retry", "done": END})
    graph.add_edge("retry", END)

    return graph.compile(checkpointer=checkpointer)
