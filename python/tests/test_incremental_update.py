"""
增量更新测试 — 文本级 diff 真增量 / 快照管理 / CDC difflib

运行: pytest tests/test_incremental_update.py -v
验证: 文档局部修改时只重新处理变化 chunk, 不触发全量重建
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.knowledge_update_agent import (  # noqa: E402
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
)
from agents.knowledge_extract_agent import (  # noqa: E402
    Entity,
    ExtractionResult,
    Relation,
)
from schema import DocType, DocumentChunk  # noqa: E402


# ── fakes ────────────────────────────────────────────────────

class MutableDocParser:
    """可变的 fake 解析器: 测试中可修改文档内容, 模拟真实文件修改"""

    def __init__(self, doc_contents: dict[str, list[str]]):
        self.doc_contents = doc_contents  # file_path → list of chunk texts
        self.parse_calls = 0

    async def parse(self, file_path):
        self.parse_calls += 1
        contents = self.doc_contents.get(file_path, [])
        doc_id = file_path.replace("/", "_")[:8]
        return [
            DocumentChunk(
                content=text,
                doc_id=doc_id,
                chunk_index=i,
                doc_type=DocType.TEXT,
                metadata={"source": file_path},
            )
            for i, text in enumerate(contents)
        ]


class TrackingVectorStore:
    def __init__(self):
        self.added_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.deleted_by_doc: list[str] = []

    async def add_chunks(self, chunks):
        ids = [c.chunk_id for c in chunks]
        self.added_ids.extend(ids)
        return len(chunks)

    async def delete_by_chunk_ids(self, chunk_ids):
        self.deleted_ids.extend(chunk_ids)
        return len(chunk_ids)

    async def delete_by_doc_id(self, doc_id):
        self.deleted_by_doc.append(doc_id)
        return 3


class TrackingExtractor:
    def __init__(self):
        self.extracted_texts: list[str] = []

    async def extract(self, chunks):
        self.extracted_texts.extend(c.content for c in chunks)
        return [ExtractionResult(
            entities=[Entity(name=f"E{len(self.extracted_texts)}", type="Concept")],
            relations=[Relation("A", "related_to", "B", 0.8)],
            events=[],
            source_chunk_id=chunks[0].chunk_id if chunks else "",
        )]


class TrackingKG:
    def __init__(self):
        self.upserted: list[str] = []
        self.relations: list[tuple] = []
        self.deleted_sources: list[str] = []
        self.deleted_chunk_ids: list[str] = []
        self.provenance: list[tuple[str, str]] = []

    async def upsert_entity(self, entity, version=1, source="", source_chunk_id=""):
        self.upserted.append(entity.name)
        self.provenance.append((source, source_chunk_id))

    async def add_relation(self, relation, source="", source_chunk_id=""):
        self.relations.append((relation.head, relation.relation, relation.tail))

    async def delete_by_source(self, source):
        self.deleted_sources.append(source)

    async def delete_by_chunk_ids(self, chunk_ids):
        self.deleted_chunk_ids.extend(chunk_ids)
        return len(chunk_ids)


def _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return KnowledgeUpdateAgent(
        doc_parser=parser,
        knowledge_extractor=extractor,
        vector_store=vs,
        knowledge_graph=kg,
    )


# ── 真增量更新 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_full_ingest(tmp_path, monkeypatch):
    """新建文档: 全量入库 + 保存快照"""
    parser = MutableDocParser({"a.md": ["chunk-0 内容", "chunk-1 内容", "chunk-2 内容"]})
    extractor, vs, kg = TrackingExtractor(), TrackingVectorStore(), TrackingKG()
    agent = _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch)

    change = DocumentChange(file_path="a.md", change_type=ChangeType.CREATED)
    result = await agent.process_change(change)

    assert result.success
    assert result.vectors_added == 3
    assert len(extractor.extracted_texts) == 3
    # 快照已持久化
    assert len(agent._snapshots) == 1


@pytest.mark.asyncio
async def test_modify_only_reprocesses_changed_chunks(tmp_path, monkeypatch):
    """修改文档: 只有变化的 chunk 被重新处理 (真增量)"""
    parser = MutableDocParser({"a.md": ["chunk-0 内容", "chunk-1 内容", "chunk-2 内容"]})
    extractor, vs, kg = TrackingExtractor(), TrackingVectorStore(), TrackingKG()
    agent = _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch)

    # 首次入库
    await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.CREATED))
    vs.added_ids.clear()
    extractor.extracted_texts.clear()

    # 修改: 只有 chunk-1 变了
    parser.doc_contents["a.md"] = ["chunk-0 内容", "chunk-1 修改后的内容", "chunk-2 内容"]
    result = await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.MODIFIED))

    assert result.success
    # 只重新向量化 1 个 chunk
    assert result.vectors_added == 1, f"只应新增 1 个向量, 实际 {result.vectors_added}"
    assert result.chunks_reprocessed == 1
    assert result.chunks_unchanged == 2
    # 旧 chunk-1 向量被删除
    assert len(vs.deleted_ids) == 1
    assert "chunk-1" in vs.deleted_ids[0]
    # 只对变化 chunk 重新抽取
    assert extractor.extracted_texts == ["chunk-1 修改后的内容"], \
        f"抽取应只覆盖变化 chunk: {extractor.extracted_texts}"
    assert len(kg.deleted_chunk_ids) == 1, "变化 chunk 的旧图谱 provenance 应被清理"
    assert kg.provenance[-1][1].endswith("chunk-1"), "图谱写入应记录来源 chunk"
    # diff 报告记录了变化
    assert result.change.diff_chunks, "diff 摘要应非空"


@pytest.mark.asyncio
async def test_modify_no_change_nothing_reprocessed(tmp_path, monkeypatch):
    """内容未变化: 0 重处理"""
    parser = MutableDocParser({"a.md": ["chunk-0 内容"]})
    extractor, vs, kg = TrackingExtractor(), TrackingVectorStore(), TrackingKG()
    agent = _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch)

    await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.CREATED))
    vs.added_ids.clear()
    extractor.extracted_texts.clear()

    result = await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.MODIFIED))
    assert result.success
    assert result.chunks_reprocessed == 0
    assert result.chunks_unchanged == 1
    assert vs.added_ids == []
    assert vs.deleted_ids == []
    assert extractor.extracted_texts == []


@pytest.mark.asyncio
async def test_modify_without_snapshot_falls_back_to_full(tmp_path, monkeypatch):
    """无快照 (如服务重启后): 退化为全量重建"""
    parser = MutableDocParser({"a.md": ["chunk-0 内容"]})
    extractor, vs, kg = TrackingExtractor(), TrackingVectorStore(), TrackingKG()
    agent = _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch)

    result = await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.MODIFIED))
    assert result.success
    assert result.chunks_reprocessed == 1
    assert result.vectors_added == 1


@pytest.mark.asyncio
async def test_delete_cleans_everything(tmp_path, monkeypatch):
    """删除文档: 向量 + 图谱 + 快照全清理"""
    parser = MutableDocParser({"a.md": ["chunk-0 内容"]})
    extractor, vs, kg = TrackingExtractor(), TrackingVectorStore(), TrackingKG()
    agent = _make_agent(parser, extractor, vs, kg, tmp_path, monkeypatch)

    await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.CREATED))
    result = await agent.process_change(DocumentChange(file_path="a.md", change_type=ChangeType.DELETED))

    assert result.success
    assert vs.deleted_by_doc, "应调用按 doc_id 删除"
    assert kg.deleted_sources == ["a.md"]
    assert agent._snapshots == {}, "快照应被移除"


# ── 更新 Agent difflib 精确 diff ─────────────────────────────

def test_cdc_compute_diff_precise():
    """difflib 行级 diff: 精确定位修改行 (非集合对比)"""
    before = "张三\n李四\n王五\n"
    after = "张三\n李四改\n王五\n"
    diff = KnowledgeUpdateAgent._compute_diff(before.splitlines(), after.splitlines())

    assert diff["added_count"] == 1
    assert diff["removed_count"] == 1
    assert diff["added_lines"] == ["李四改"]
    assert diff["removed_lines"] == ["李四"]
    assert diff["changed_blocks"] == 1
    assert diff["change_ratio"] == round(2 / 6, 4)  # 2 行变化 / 6 行总量 (保留 4 位)
    # 小文档单行修改即超过 30% 阈值, 属预期语义
    assert diff["is_major_change"] is True


def test_cdc_compute_diff_insert_and_delete():
    before = "a\nb\nc"
    after = "a\nc\nd\ne"
    diff = KnowledgeUpdateAgent._compute_diff(before.splitlines(), after.splitlines())

    assert diff["removed_lines"] == ["b"]
    assert diff["added_lines"] == ["d", "e"]
    assert diff["change_ratio"] > 0


def test_cdc_compute_diff_identical():
    diff = KnowledgeUpdateAgent._compute_diff(["相同内容"], ["相同内容"])
    assert diff["added_count"] == 0
    assert diff["removed_count"] == 0
    assert diff["change_ratio"] == 0.0


def test_update_agent_compute_diff_summary():
    """知识更新 Agent 的 diff 摘要"""
    agent = KnowledgeUpdateAgent()
    report = agent._compute_diff(
        ["第一段内容。", "第二段内容。"],
        ["第一段内容。", "第二段修改后的内容。", "新增第三段。"],
    )
    # difflib 把"修改"拆为 删1行(旧) + 增2行(改后行 + 新行)
    assert report["added_count"] == 2
    assert report["removed_count"] == 1
    assert "+ 新增第三段。" in report["changed_lines_summary"]
    assert "- 第二段内容。" in report["changed_lines_summary"]
