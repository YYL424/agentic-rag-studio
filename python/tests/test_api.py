"""
API 层测试 — SSE 流式输出 / HITL 审核端点 / 健康检查

运行: pytest tests/test_api.py -v
全部用 fake graph, 不调用真实 LLM / 外部服务
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_fake_streaming_qa_graph():
    """构造一个会流式输出 token 的 fake QA graph"""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langgraph.graph import END, StateGraph

    from agents.qa_agent import QAResult, QueryIntent

    class FakeStreamingChatModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            for tok in ["流", "式", "输", "出", "！"]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=tok))

    from typing import TypedDict

    class QAState(TypedDict, total=False):
        question: str
        result: object

    async def answer_node(state):
        model = FakeStreamingChatModel()
        collected: list[str] = []
        async for chunk in model.astream(
            [HumanMessage(content="hi")],
            config={"tags": ["final_answer"]},
        ):
            collected.append(chunk.content)
        return {"result": QAResult(
            question=state.get("question", ""),
            answer="".join(collected),
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.85,
            reasoning_steps=["fake"],
            retrieval_rounds=1,
        )}

    graph = StateGraph(QAState)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    from langgraph.checkpoint.memory import InMemorySaver
    return graph.compile(checkpointer=InMemorySaver())


def _build_fake_hitl_ingest_graph():
    """fake ingest graph (HITL 开启): 复用 Phase 2 测试的 fakes"""
    from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
    from langgraph.checkpoint.memory import InMemorySaver

    from orchestrator.graph import _build_ingest_graph
    from schema import DocType, DocumentChunk

    class FakeParser:
        async def parse_batch(self, file_paths):
            return [
                DocumentChunk(content="张三担任腾讯公司CEO。", doc_id="d1", chunk_index=0,
                              doc_type=DocType.TEXT, metadata={"source": fp})
                for fp in file_paths
            ]

    class FakeExtractor:
        async def extract(self, chunks):
            return [ExtractionResult(
                entities=[Entity(name="张三", type="Person")],
                relations=[Relation("张三", "works_at", "腾讯", 0.9)],
                events=[],
            )]

    class FakeVS:
        async def add_chunks(self, chunks):
            return len(chunks)

    class FakeKG:
        async def upsert_entity(self, entity, version=1, source="", source_chunk_id=""):
            pass

        async def add_relation(self, relation, source="", source_chunk_id=""):
            pass

    return _build_ingest_graph(
        FakeParser(), FakeExtractor(), FakeVS(), FakeKG(),
        checkpointer=InMemorySaver(), enable_hitl=True,
    )


@pytest.fixture
def client(monkeypatch):
    """TestClient + lifespan 启动 (外部服务初始化失败会被优雅降级)"""
    import api.main as api_mod

    with pytest.MonkeyPatch.context() as mp:
        # 阻止 lifespan 尝试连接真实 chroma/neo4j
        async def noop_init(self):
            return None
        async def noop_close(self):
            return None
        from services.vector_store import VectorStoreService
        from services.knowledge_graph import KnowledgeGraphService
        mp.setattr(VectorStoreService, "init", noop_init)
        mp.setattr(KnowledgeGraphService, "init", noop_init)
        mp.setattr(KnowledgeGraphService, "close", noop_close)

        from fastapi.testclient import TestClient
        with TestClient(api_mod.app) as c:
            yield c, api_mod


# ── SSE 流式问答 ─────────────────────────────────────────────

def test_sse_stream_output(client):
    """SSE 端点: token 逐块输出 + final 事件 + [DONE]"""
    c, api_mod = client
    api_mod.workflows["qa"] = _build_fake_streaming_qa_graph()

    with c.stream("POST", "/api/qa/ask_stream", json={"question": "测试"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events: list[dict] = []
        done_seen = False
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done_seen = True
                break
            events.append(json.loads(payload))

    tokens = [e for e in events if e.get("type") == "token"]
    finals = [e for e in events if e.get("type") == "final"]
    assert [t["content"] for t in tokens] == ["流", "式", "输", "出", "！"], f"token 顺序错误: {tokens}"
    assert done_seen, "应以 [DONE] 结束"
    assert finals, "应有 final 元信息事件"
    assert finals[0]["confidence"] == 0.85


def test_ask_endpoint_sync(client):
    """同步问答端点"""
    c, api_mod = client
    api_mod.workflows["qa"] = _build_fake_streaming_qa_graph()

    resp = c.post("/api/qa/ask", json={"question": "测试"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "流式输出！"
    assert data["retrieval_rounds"] == 1


# ── HITL 上传 + 审核流程 ─────────────────────────────────────

def test_ingest_upload_requires_review_then_approve(client):
    """上传 → review_required → 审核通过 → 入库完成"""
    c, api_mod = client
    api_mod.workflows["ingest"] = _build_fake_hitl_ingest_graph()

    # 上传
    resp = c.post("/api/ingest/upload", files={"file": ("test.md", b"# test", "text/markdown")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "review_required", f"HITL 模式应返回待审核: {data}"
    assert re.fullmatch(r"[0-9a-f]{32}\.md", data["file_id"])
    assert data["thread_id"].startswith("ingest-")
    assert data["pending_review"], "应返回待审核的实体/关系摘要"

    # 审核通过
    resp2 = c.post("/api/ingest/review", json={"thread_id": data["thread_id"], "approved": True})
    assert resp2.status_code == 200
    final = resp2.json()
    assert final["status"] == "approved"
    assert final["vectors_stored"] == 1
    assert final["entities_stored"] == 1


def test_ingest_review_reject(client):
    """审核驳回: 不入库"""
    c, api_mod = client
    api_mod.workflows["ingest"] = _build_fake_hitl_ingest_graph()

    resp = c.post("/api/ingest/upload", files={"file": ("test.md", b"# test", "text/markdown")})
    thread_id = resp.json()["thread_id"]

    resp2 = c.post("/api/ingest/review", json={"thread_id": thread_id, "approved": False})
    assert resp2.status_code == 200
    final = resp2.json()
    assert final["status"] == "rejected"
    assert final["entities_stored"] == 0


def test_admin_delete_removes_managed_upload(client, tmp_path, monkeypatch):
    """删除更新成功后应清理上传文件，且仅允许 upload_dir 内路径。"""
    from types import SimpleNamespace

    c, api_mod = client
    monkeypatch.setattr(api_mod.settings, "upload_dir", str(tmp_path))
    managed_file = tmp_path / "managed.md"
    managed_file.write_text("test", encoding="utf-8")

    class FakeUpdateWorkflow:
        async def ainvoke(self, state, config):
            change = state["changes"][0]
            return {"results": [SimpleNamespace(
                change=change,
                vectors_added=0,
                vectors_deleted=2,
                entities_added=0,
                relations_added=0,
                chunks_reprocessed=0,
                chunks_unchanged=0,
                success=True,
                processing_time_ms=1.0,
            )]}

    api_mod.workflows["update"] = FakeUpdateWorkflow()
    resp = c.post("/api/admin/update", json={"file_path": "managed.md", "change_type": "deleted"})

    assert resp.status_code == 200
    assert resp.json()["vectors_deleted"] == 2
    assert not managed_file.exists()


# ── 健康检查 ─────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    resp = c.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"] == "2.0.0"


def test_readiness_reports_dependency_failure(client):
    c, api_mod = client
    api_mod.app.state.dependency_status = {
        "vector_store": True,
        "knowledge_graph": False,
    }

    health_resp = c.get("/api/health")
    ready_resp = c.get("/api/health/ready")

    assert health_resp.json()["status"] == "degraded"
    assert ready_resp.status_code == 503


def test_root_html(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "AgentKnowledgeHub" in resp.text
    assert '<label class="upload-zone" id="uploadZone" for="fileInput">' in resp.text
    # CHAT_HTML 必须保留 JS 转义；Python 若把 \n 展开为真实换行，会让整段脚本语法错误。
    assert "已入库\\n分块" in resp.text
    assert "已入库\n分块" not in resp.text
    assert "buf.indexOf('\\n\\n')" in resp.text
    assert "⏳ 解析中" in resp.text
    assert "原因: ' + detail" in resp.text
    assert "e.target.value = ''" in resp.text
