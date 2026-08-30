"""
LangGraph 编排层测试 — Checkpointer / HITL interrupt / 结构化输出 / 实体消解

运行: pytest tests/test_orchestrator.py -v
全部使用 fake agents, 不调用真实 LLM / 外部服务
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.knowledge_extract_agent import (  # noqa: E402
    Entity,
    ExtractionResult,
    KnowledgeEvent,
    KnowledgeExtractAgent,
    Relation,
)
from orchestrator.graph import (  # noqa: E402
    _build_ingest_graph,
    create_checkpointer,
)
from schema import DocType, DocumentChunk  # noqa: E402


# ── fakes ────────────────────────────────────────────────────

class FakeDocParser:
    async def parse_batch(self, file_paths):
        return [
            DocumentChunk(
                content="张三担任腾讯公司CEO。",
                doc_id="d1",
                chunk_index=0,
                doc_type=DocType.TEXT,
                metadata={"source": fp, "heading_path": ""},
            )
            for fp in file_paths
        ]


class FakeExtractor:
    def __init__(self):
        self.extract_calls = 0

    async def extract(self, chunks):
        self.extract_calls += 1
        return [
            ExtractionResult(
                entities=[Entity(name="张三", type="Person", description="CEO")],
                relations=[Relation(head="张三", relation="works_at", tail="腾讯", confidence=0.9)],
                events=[KnowledgeEvent(trigger="担任", type="任命", participants=["张三"])],
                source_chunk_id=chunks[0].chunk_id if chunks else "",
            )
        ]


class FakeVectorStore:
    def __init__(self):
        self.added = 0

    async def add_chunks(self, chunks):
        self.added += len(chunks)
        return len(chunks)


class FakeKnowledgeGraph:
    def __init__(self):
        self.entities = []
        self.relations = []

    async def upsert_entity(self, entity, version=1, source="", source_chunk_id=""):
        self.entities.append(entity.name)

    async def add_relation(self, relation, source="", source_chunk_id=""):
        self.relations.append((relation.head, relation.relation, relation.tail))


# ── checkpointer ─────────────────────────────────────────────

def test_create_checkpointer_default():
    """默认返回内存 Checkpointer"""
    cp = create_checkpointer(backend="memory")
    assert cp is not None


def test_create_checkpointer_invalid_backend_falls_back():
    """无效后端配置自动降级, 不抛异常"""
    cp = create_checkpointer(backend="nonexistent_backend")
    assert cp is not None


# ── ingest graph (no HITL) ───────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_flow_without_hitl():
    """无 HITL: 一条流水线跑通 parse → extract → store"""
    parser, extractor = FakeDocParser(), FakeExtractor()
    vs, kg = FakeVectorStore(), FakeKnowledgeGraph()

    graph = _build_ingest_graph(parser, extractor, vs, kg, checkpointer=create_checkpointer("memory"), enable_hitl=False)
    # 编译了 checkpointer 的图必须携带 thread_id
    config = {"configurable": {"thread_id": "no-hitl-test"}}
    result = await graph.ainvoke({"file_paths": ["a.md", "b.md"]}, config=config)

    assert len(result["chunks"]) == 2
    assert result["vectors_stored"] == 2
    assert result["entities_stored"] == 1
    assert kg.entities == ["张三"]
    assert kg.relations == [("张三", "works_at", "腾讯")]
    assert "__interrupt__" not in result


# ── ingest graph (HITL) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume():
    """HITL: 抽取后流程挂起, 审核通过后恢复入库"""
    parser, extractor = FakeDocParser(), FakeExtractor()
    vs, kg = FakeVectorStore(), FakeKnowledgeGraph()

    graph = _build_ingest_graph(parser, extractor, vs, kg, checkpointer=create_checkpointer("memory"), enable_hitl=True)
    config = {"configurable": {"thread_id": "hitl-test-1"}}

    # 第一次 ainvoke: 在 review 节点挂起
    result = await graph.ainvoke({"file_paths": ["a.md"]}, config=config)
    assert "__interrupt__" in result, "HITL 模式应在 review 节点产生 interrupt"
    assert vs.added == 0, "审核未通过前不应入库"
    assert kg.entities == [], "审核未通过前不应写图谱"

    # 恢复: 审核通过 (langgraph 1.2.x 中 resume 值需为非 None, 用 {"approved": True} 表示通过)
    from langgraph.types import Command
    final = await graph.ainvoke(Command(resume={"approved": True}), config=config)
    assert final["vectors_stored"] == 1
    assert final["entities_stored"] == 1
    assert final["review_note"] == "approved"


@pytest.mark.asyncio
async def test_hitl_reject():
    """HITL: 审核驳回 → 知识不入库"""
    parser, extractor = FakeDocParser(), FakeExtractor()
    vs, kg = FakeVectorStore(), FakeKnowledgeGraph()

    graph = _build_ingest_graph(parser, extractor, vs, kg, checkpointer=create_checkpointer("memory"), enable_hitl=True)
    config = {"configurable": {"thread_id": "hitl-test-2"}}

    await graph.ainvoke({"file_paths": ["a.md"]}, config=config)

    from langgraph.types import Command
    final = await graph.ainvoke(Command(resume={"approved": False}), config=config)
    assert final["review_note"] == "rejected"
    assert final["vectors_stored"] == 0, "驳回后不应写入向量库"
    assert vs.added == 0, "驳回后向量库必须保持不变"
    assert final["entities_stored"] == 0, "驳回后不应写图谱实体"
    assert kg.entities == []


# ── structured output conversion ─────────────────────────────

def test_to_extraction_result_from_pydantic():
    """Pydantic ExtractionOutput → 系统 dataclass"""
    from agents.knowledge_extract_agent import ExtractionOutput

    output = ExtractionOutput(
        entities=[{"name": "张三", "type": "Person", "description": "CEO"}],
        relations=[{"head": "张三", "relation": "works_at", "tail": "腾讯", "confidence": 0.9}],
        events=[{"trigger": "担任", "type": "任命", "participants": ["张三"]}],
    )
    result = KnowledgeExtractAgent._to_extraction_result(output, "chunk-1")

    assert result.source_chunk_id == "chunk-1"
    assert result.entities[0].name == "张三"
    assert result.entities[0].type == "Person"
    assert result.relations[0].confidence == 0.9
    assert result.events[0].trigger == "担任"


def test_parse_response_fallback_json():
    """JSON 字符串降级解析路径"""
    raw = '{"entities": [{"name": "李四", "type": "Person", "description": ""}], "relations": [], "events": []}'
    result = KnowledgeExtractAgent()._parse_response(raw, "c1")
    assert len(result.entities) == 1
    assert result.entities[0].name == "李四"


def test_parse_response_code_fence():
    """带 ``` 代码围栏的 JSON 也能解析"""
    raw = '```json\n{"entities": [], "relations": [], "events": []}\n```'
    result = KnowledgeExtractAgent()._parse_response(raw, "c1")
    assert result.entities == []


@pytest.mark.asyncio
async def test_deepseek_uses_json_mode_with_explicit_schema(monkeypatch):
    """DeepSeek thinking 模式不应尝试 tool_choice，json_mode prompt 必须携带完整 Schema。"""
    from agents.knowledge_extract_agent import ExtractionOutput
    from config import settings

    calls = []

    class FakeStructuredLLM:
        async def ainvoke(self, messages):
            system = next(message for message in messages if isinstance(message, SystemMessage))
            calls.append(system.content)
            return ExtractionOutput()

    class FakeLLM:
        def with_structured_output(self, schema, method):
            calls.append(method)
            return FakeStructuredLLM()

    monkeypatch.setattr(settings, "structured_output_method", "auto")
    monkeypatch.setattr(settings, "openai_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "openai_model", "deepseek-reasoner")
    agent = KnowledgeExtractAgent()
    agent.llm = FakeLLM()

    result = await agent._structured_extract("测试文本")

    assert isinstance(result, ExtractionOutput)
    assert calls[0] == "json_mode"
    assert "JSON Schema" in calls[1]
    assert '"entities"' in calls[1]
    assert "function_calling" not in calls


def test_deduplicate():
    """chunk 内去重，但保留跨 chunk provenance"""
    agent = KnowledgeExtractAgent()
    r1 = ExtractionResult(
        entities=[Entity("张三", "Person"), Entity("张三", "Person")],
        relations=[],
        events=[],
        source_chunk_id="c1",
    )
    r2 = ExtractionResult(entities=[Entity("张三", "Person")], relations=[], events=[])
    deduped = agent._deduplicate([r1, r2])
    assert len(deduped[0].entities) == 1
    assert len(deduped[1].entities) == 1, "跨 chunk 出现必须保留，以记录 provenance"


# ── entity resolution ────────────────────────────────────────

@pytest.mark.asyncio
async def test_entity_resolver_normalization_merge():
    """归一化合并: 腾讯科技有限公司 → 腾讯科技"""
    from services.entity_resolver import EntityResolver

    resolver = EntityResolver(model_name=None)  # 无模型 → 纯归一化
    results = [
        ExtractionResult(
            entities=[Entity("腾讯科技", "Organization")],
            relations=[Relation("张三", "works_at", "腾讯科技", 0.9)],
            events=[],
        ),
        ExtractionResult(
            entities=[Entity("腾讯科技有限公司", "Organization")],
            relations=[Relation("李四", "works_at", "腾讯科技有限公司", 0.9)],
            events=[],
        ),
    ]
    resolved = await resolver.resolve(results)

    names = [e.name for r in resolved for e in r.entities]
    assert len(set(names)) == 1, f"同指实体应合并: {names}"
    # 关系头尾也应替换为规范名
    tails = [r.tail for r in resolved[1].relations]
    assert all(t == "腾讯科技" for t in tails)


@pytest.mark.asyncio
async def test_entity_resolver_no_model_no_crash():
    """无 embedding 模型时优雅降级, 不抛异常"""
    from services.entity_resolver import EntityResolver

    resolver = EntityResolver(model_name="nonexistent-model-name")
    results = [ExtractionResult(entities=[Entity("完全不同实体A", "Concept")], relations=[], events=[])]
    resolved = await resolver.resolve(results)
    assert resolved[0].entities[0].name == "完全不同实体A"
