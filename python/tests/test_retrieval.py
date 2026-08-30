"""
检索层测试 — Qdrant 嵌入式模式 / Reranker 降级 / 混合排序 / Self-RAG 循环

运行: pytest tests/test_retrieval.py -v
全部离线: Qdrant 嵌入式本地模式, 无需 Docker / 外部服务 / 真实 LLM 调用
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import DocType, DocumentChunk  # noqa: E402


class FakeEmbeddings:
    """确定性 4 维 fake embeddings"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        # 简单哈希向量: 相似文本 → 相似向量
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h[:4]]


def _make_chunk(content: str, doc_id: str, idx: int) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        doc_id=doc_id,
        chunk_index=idx,
        doc_type=DocType.TEXT,
        metadata={"source": f"doc{doc_id}.md", "heading_path": "测试", "is_table": False, "is_image": False},
    )


# ── Qdrant 嵌入式本地模式 (无需 Docker) ─────────────────────

@pytest_asyncio.fixture
async def qdrant_store(tmp_path, monkeypatch):
    from config import settings
    from services.vector_store import VectorStoreService

    monkeypatch.setattr(settings, "vector_store_type", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant"))
    store = VectorStoreService(embeddings=FakeEmbeddings())
    yield store
    await store.close()


# ── Chroma 本地持久化模式 (无需 chroma server) ──────────────

@pytest_asyncio.fixture
async def chroma_store(tmp_path, monkeypatch):
    from config import settings
    from services.vector_store import VectorStoreService

    monkeypatch.setattr(settings, "vector_store_type", "chroma")
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    store = VectorStoreService(embeddings=FakeEmbeddings())
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_chroma_add_search_delete_roundtrip(chroma_store):
    """Chroma 本地模式: 写入 → 检索 → 删除 (与 Qdrant 同一接口)"""
    await chroma_store.init()

    chunks = [
        _make_chunk("张三担任腾讯公司CEO，负责微信事业部。", "docA", 0),
        _make_chunk("李四负责市场部，专注海外推广。", "docA", 1),
        _make_chunk("王五负责财务部，管理公司预算。", "docB", 0),
    ]
    added = await chroma_store.add_chunks(chunks)
    assert added == 3

    results = await chroma_store.search("张三担任腾讯公司CEO，负责微信事业部。", top_k=3)
    assert len(results) == 3
    top_content = results[0][0]["content"]
    assert "张三" in top_content, f"最相关 chunk 应命中张三: {top_content}"

    # 元数据应保留
    assert results[0][0]["metadata"]["doc_id"] == "docA"
    assert results[0][0]["metadata"]["heading_path"] == "测试"

    # 按 doc_id 删除
    deleted = await chroma_store.delete_by_doc_id("docA")
    assert deleted == 2
    remaining = await chroma_store.search("公司预算", top_k=5)
    assert all("王五" in r[0]["content"] for r in remaining)

    # 按 chunk_id 精确删除
    deleted = await chroma_store.delete_by_chunk_ids(["docB#chunk-0"])
    assert deleted == 1
    stats = await chroma_store.get_stats()
    assert stats["backend"] == "chroma"
    assert stats["total_vectors"] == 0


@pytest.mark.asyncio
async def test_qdrant_add_search_delete_roundtrip(qdrant_store):
    """Qdrant: 写入 → 检索 → 按 doc_id 删除 → 按 chunk_id 删除"""
    await qdrant_store.init()

    chunks = [
        _make_chunk("张三担任腾讯公司CEO，负责微信事业部。", "docA", 0),
        _make_chunk("李四负责市场部，专注海外推广。", "docA", 1),
        _make_chunk("王五负责财务部，管理公司预算。", "docB", 0),
    ]
    added = await qdrant_store.add_chunks(chunks)
    assert added == 3

    # 检索: 与"张三职位"相关的 chunk 应排第一
    results = await qdrant_store.search("张三担任腾讯公司CEO，负责微信事业部。", top_k=3)
    assert len(results) == 3
    top_content = results[0][0]["content"]
    assert "张三" in top_content, f"最相关 chunk 应命中张三: {top_content}"
    assert results[0][1] > results[2][1], "分数应降序"

    # 元数据应保留
    assert results[0][0]["metadata"]["doc_id"] == "docA"
    assert results[0][0]["metadata"]["heading_path"] == "测试"

    # 按 doc_id 删除
    deleted = await qdrant_store.delete_by_doc_id("docA")
    assert deleted == 2
    remaining = await qdrant_store.search("公司预算", top_k=5)
    assert all("王五" in r[0]["content"] for r in remaining)

    # 按 chunk_id 精确删除
    deleted = await qdrant_store.delete_by_chunk_ids(["docB#chunk-0"])
    assert deleted == 1
    stats = await qdrant_store.get_stats()
    assert stats["backend"] == "qdrant"
    assert stats["total_vectors"] == 0
    assert stats["mode"] == "embedded"


@pytest.mark.asyncio
async def test_qdrant_get_stats_before_init(tmp_path, monkeypatch):
    """未初始化时 get_stats 优雅降级"""
    from config import settings
    from services.vector_store import VectorStoreService

    monkeypatch.setattr(settings, "vector_store_type", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant2"))
    store = VectorStoreService(embeddings=FakeEmbeddings())
    stats = await store.get_stats()
    assert stats["status"] == "not_initialized"


@pytest.mark.asyncio
async def test_qdrant_rejects_embedding_dimension_mismatch(tmp_path, monkeypatch):
    """切换 embedding 模型时不应把不同维度的向量写进已有 collection。"""
    from config import settings
    from services.vector_store import VectorStoreService

    qdrant_path = str(tmp_path / "qdrant-dimension")
    monkeypatch.setattr(settings, "vector_store_type", "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", qdrant_path)

    original = VectorStoreService(embeddings=FakeEmbeddings())
    await original.init()
    await original.close()

    class ThreeDimEmbeddings(FakeEmbeddings):
        @staticmethod
        def _vec(_text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    incompatible = VectorStoreService(embeddings=ThreeDimEmbeddings())
    with pytest.raises(RuntimeError, match="维度不兼容"):
        await incompatible.init()
    await incompatible.close()


# ── Reranker ─────────────────────────────────────────────────

def test_reranker_unavailable_falls_back_gracefully(monkeypatch):
    """模型不可用: available=False, rerank 返回 None (调用方降级)"""
    import sys
    from services.reranker import RerankerService

    # 直接模拟可选依赖缺失，避免测试加载数 GB 模型或触发本机原生库兼容问题。
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    service = RerankerService(model_name="nonexistent-model")
    assert service.available is False
    assert service.rerank("query", ["doc1", "doc2"]) is None


def test_reranker_score_normalization():
    """sigmoid 归一化: 分数映射到 0-1"""
    from services.reranker import RerankerService

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [2.0, 0.0, -2.0]

    service = RerankerService(model_name="fake")
    service._model = FakeCrossEncoder()  # 注入 fake 模型
    scores = service.rerank("q", ["a", "b", "c"])

    assert scores is not None
    assert scores[0] > 0.8, f"sigmoid(2.0)≈0.88, got {scores[0]}"
    assert abs(scores[1] - 0.5) < 0.01
    assert scores[2] < 0.2, f"sigmoid(-2.0)≈0.12, got {scores[2]}"


# ── QAAgent 混合排序 / Self-RAG ─────────────────────────────

def test_hybrid_rerank_weighting():
    """图谱结果权重 1.2 > 向量 1.0"""
    from agents.qa_agent import QAAgent, RetrievedContext

    v = RetrievedContext(content="向量内容", source="vs", score=0.9, retrieval_type="vector")
    g = RetrievedContext(content="图谱内容", source="kg", score=0.8, retrieval_type="graph")
    ranked = QAAgent._hybrid_rerank([v, g])
    assert ranked[0].content == "图谱内容", f"图谱加权后应反超: {[ (r.content, r.score) for r in ranked ]}"


def test_dedup_by_content():
    from agents.qa_agent import QAAgent, RetrievedContext
    c1 = RetrievedContext(content="相同内容前缀" * 10, source="a", score=0.9, retrieval_type="vector")
    c2 = RetrievedContext(content="相同内容前缀" * 10, source="b", score=0.8, retrieval_type="graph")
    result = QAAgent._dedup([c1, c2])
    assert len(result) == 1


@pytest.mark.asyncio
async def test_graph_retrieval_uses_parameterized_templates_only():
    """图检索只调用受控模板，不触碰任意 Cypher 执行入口。"""
    from agents.qa_agent import QAAgent

    class FakeGraph:
        def __init__(self):
            self.neighbor_calls = []
            self.path_calls = []

        async def get_neighbors(self, entity, hops):
            self.neighbor_calls.append((entity, hops))
            return [{
                "source": entity,
                "relations": ["WORKS_AT"],
                "target": "腾讯",
                "target_type": "Organization",
                "target_desc": "公司",
            }]

        async def find_paths(self, name_a, name_b, max_hops):
            self.path_calls.append((name_a, name_b, max_hops))
            return [{"node_names": [name_a, name_b], "rel_types": ["WORKS_AT"]}]

        async def execute_cypher(self, *_args, **_kwargs):
            raise AssertionError("QA 不应执行任意 Cypher")

    graph = FakeGraph()
    agent = object.__new__(QAAgent)
    agent.vector_store = None
    agent.knowledge_graph = graph
    agent.reranker = None
    contexts = await agent._graph_retrieve("忽略", {"entities": ["张三", "腾讯"]})

    assert graph.neighbor_calls == [("张三", 2), ("腾讯", 2)]
    assert graph.path_calls == [("张三", "腾讯", 5)]
    assert any(c.metadata.get("strategy") == "shortest_path" for c in contexts)


@pytest.mark.asyncio
async def test_vector_and_graph_retrieval_start_concurrently():
    """两个检索分支必须都启动后才能互相放行，串行实现会超时。"""
    import asyncio
    from agents.qa_agent import QAAgent

    vector_started = asyncio.Event()
    graph_started = asyncio.Event()

    class FakeStore:
        async def search(self, _query, top_k):
            assert top_k == 5
            vector_started.set()
            await asyncio.wait_for(graph_started.wait(), timeout=0.5)
            return [({"content": "向量证据", "source": "doc.md"}, 0.8)]

    class FakeGraph:
        async def get_neighbors(self, entity, hops):
            assert (entity, hops) == ("张三", 2)
            graph_started.set()
            await asyncio.wait_for(vector_started.wait(), timeout=0.5)
            return []

        async def find_paths(self, *_args, **_kwargs):
            return []

    agent = object.__new__(QAAgent)
    agent.vector_store = FakeStore()
    agent.knowledge_graph = FakeGraph()
    agent.reranker = None
    contexts = await asyncio.wait_for(
        agent.retrieve("张三是谁", {"queries": ["张三是谁"], "entities": ["张三"]}),
        timeout=1,
    )
    assert contexts[0].content == "向量证据"


@pytest.mark.asyncio
async def test_self_rag_loop_rewrites_when_relevance_low(monkeypatch):
    """Self-RAG: 相关性低于阈值 → 改写查询重检"""
    from agents.qa_agent import QAAgent, QAResult, QueryIntent, RelevanceScore, RetrievedContext

    class FakeStore:
        def __init__(self):
            self.calls = 0

        async def search(self, query, top_k=5):
            self.calls += 1
            return [({"content": f"内容-{self.calls}: {query[:20]}", "source": "s", "metadata": {}}, 0.9)]

    class FakeReranker:
        available = False
        async def arerank(self, q, docs):
            return None

    agent = QAAgent(vector_store=FakeStore(), knowledge_graph=None, reranker=FakeReranker())
    agent.reranker = None

    relevance_values = iter([RelevanceScore(score=0.3), RelevanceScore(score=0.9)])

    async def fake_intent(q):
        return QueryIntent.FACTOID

    async def fake_rewrite(q, contexts=None):
        return {"queries": [q], "entities": [], "keywords": []}

    async def fake_eval(q, ctxs):
        return next(relevance_values)

    async def fake_generate(q, ctxs, intent):
        return "答案", ["done"]

    monkeypatch.setattr(agent, "_classify_intent", fake_intent)
    monkeypatch.setattr(agent, "_rewrite_query", fake_rewrite)
    monkeypatch.setattr(agent, "_rewrite_query_self_rag", fake_rewrite)
    monkeypatch.setattr(agent, "_evaluate_relevance", fake_eval)
    monkeypatch.setattr(agent, "_generate_answer", fake_generate)

    from config import settings
    monkeypatch.setattr(settings, "enable_self_rag", True)
    monkeypatch.setattr(settings, "self_rag_threshold", 0.6)
    monkeypatch.setattr(settings, "self_rag_max_rounds", 2)

    result = await agent.answer("张三的职位是什么？")
    assert result.retrieval_rounds == 2, f"相关性不足应触发重检: {result.retrieval_rounds}"
    assert "重检" in " ".join(result.reasoning_steps) or result.retrieval_rounds > 1
    # 第二轮检索合并了新旧上下文
    assert len(result.contexts) >= 1


@pytest.mark.asyncio
async def test_self_rag_skips_when_relevance_high(monkeypatch):
    """Self-RAG: 相关性充足 → 单轮直答"""
    from agents.qa_agent import QAAgent, QueryIntent, RelevanceScore

    class FakeStore:
        async def search(self, query, top_k=5):
            return [({"content": "高质量内容", "source": "s", "metadata": {}}, 0.9)]

    agent = QAAgent(vector_store=FakeStore(), knowledge_graph=None, reranker=None)

    async def fake_intent(q):
        return QueryIntent.FACTOID

    async def fake_rewrite(q):
        return {"queries": [q], "entities": [], "keywords": []}

    async def fake_eval(q, ctxs):
        return RelevanceScore(score=0.95)

    async def fake_generate(q, ctxs, intent):
        return "答案", ["done"]

    monkeypatch.setattr(agent, "_classify_intent", fake_intent)
    monkeypatch.setattr(agent, "_rewrite_query", fake_rewrite)
    monkeypatch.setattr(agent, "_evaluate_relevance", fake_eval)
    monkeypatch.setattr(agent, "_generate_answer", fake_generate)

    from config import settings
    monkeypatch.setattr(settings, "enable_self_rag", True)
    monkeypatch.setattr(settings, "self_rag_threshold", 0.6)

    result = await agent.answer("问题")
    assert result.retrieval_rounds == 1
