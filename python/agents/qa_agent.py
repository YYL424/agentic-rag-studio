"""
问答 Agent — 混合检索 (Vector + Graph) + Reranker 精排 + Self-RAG + 答案生成

核心能力:
  1. 意图识别 & 查询改写 (结构化输出)
  2. 向量检索 (语义相似度) + 参数化图谱检索
  3. BGE-Reranker 精排 (替代硬编码权重)
  4. Self-RAG 自适应检索: LLM 评估相关性, 不足则改写查询重检
  5. 基于检索结果的答案生成（带引用来源）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from services.reranker import RerankerService
from services.structured_llm import StructuredLLMAdapter

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTOID = "factoid"           # 事实型问题
    ANALYTICAL = "analytical"     # 分析型问题
    COMPARATIVE = "comparative"   # 对比型问题
    PROCEDURAL = "procedural"     # 流程型问题
    EXPLORATORY = "exploratory"   # 探索型问题


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # "vector" | "graph" | "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "retrieval_type": self.retrieval_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievedContext":
        return cls(
            content=str(value.get("content", "")),
            source=str(value.get("source", "")),
            score=float(value.get("score", 0.0)),
            retrieval_type=str(value.get("retrieval_type", "vector")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)
    retrieval_rounds: int = 1

    @property
    def retrieval_support(self) -> float:
        """Semantic name for the legacy ``confidence`` response field."""
        return self.confidence


# ── Pydantic Schema (结构化输出契约) ────────────────────────

class IntentOutput(BaseModel):
    intent: Literal["factoid", "analytical", "comparative", "procedural", "exploratory"]


class QueryRewriteOutput(BaseModel):
    queries: list[str] = Field(description="1-3 个改写后的检索查询")
    entities: list[str] = Field(description="提取的核心实体")
    keywords: list[str] = Field(description="提取的关键词")


class RelevanceScore(BaseModel):
    """Self-RAG 检索质量评估"""
    score: float = Field(ge=0, le=1, description="检索结果与问题的相关性 0-1")
    reasoning: str = Field(default="", description="评分理由")


INTENT_PROMPT = """\
你是一个查询意图分类器。根据用户问题，返回意图类别：
- factoid: 事实型（谁/什么/哪里/何时）
- analytical: 分析型（为什么/怎么理解）
- comparative: 对比型（A和B有什么区别）
- procedural: 流程型（怎么做/步骤）
- exploratory: 探索型（有哪些/概述）
"""

QUERY_REWRITE_PROMPT = """\
你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
"""

ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 每个可验证结论后使用 [1]、[2] 形式引用对应证据编号
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
6. 不要引用未提供的证据编号
"""

# Self-RAG: 评估检索质量
RELEVANCE_PROMPT = """\
你是一个检索质量评估器。评估以下检索结果与用户问题的相关性。
0-1 打分: 1.0 表示检索结果完全回答了问题, 0 表示完全无关。
"""


class QAAgent:
    """
    问答 Agent

    工作流:
      query → intent_classify → rewrite → parallel_retrieve
            → hybrid_rerank → bge_rerank → [Self-RAG loop] → generate_answer
    """

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        reranker: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

        # Reranker: 传入 None 时按 settings 决定是否启用
        if reranker is None and settings.enable_reranker:
            reranker = RerankerService()
        self.reranker = reranker

        # 统一供应商适配；DeepSeek 避免不兼容的 tool_choice，失败时由各节点降级。
        self.structured_llm = StructuredLLMAdapter(self.llm)

    # ── public API ───────────────────────────────────────────

    async def answer(self, question: str) -> QAResult:
        """完整问答流程 (Self-RAG + Reranker)"""
        # Both calls depend only on the question. Starting them together removes
        # one model round-trip from the critical path without changing call count.
        intent, rewritten = await asyncio.gather(
            self._classify_intent(question),
            self._rewrite_query(question),
        )
        all_contexts = await self.retrieve(question, rewritten, top_k=8)
        all_contexts, rounds, reasoning_steps = await self.refine_retrieval(
            question, all_contexts, rounds=1,
        )
        reasoning_steps.insert(0, f"识别问题意图: {intent.value}")
        return await self.build_result(question, intent, all_contexts, rounds, reasoning_steps)

    async def refine_retrieval(
        self,
        question: str,
        contexts: list[RetrievedContext],
        *,
        rounds: int = 1,
    ) -> tuple[list[RetrievedContext], int, list[str]]:
        """Run the bounded Self-RAG loop as a resumable workflow stage."""
        reasoning_steps = [f"第 {rounds} 轮检索到 {len(contexts)} 条上下文"]
        if not settings.enable_self_rag:
            return contexts, rounds, reasoning_steps

        current = contexts
        while rounds < settings.self_rag_max_rounds:
            relevance = await self._evaluate_relevance(question, current) if current else None
            if current and (relevance is None or relevance.score >= settings.self_rag_threshold):
                break
            score_text = f"{relevance.score:.2f}" if relevance is not None else "无结果"
            rounds += 1
            reasoning_steps.append(
                f"检索相关性 {score_text}，改写查询重检（第 {rounds} 轮）"
            )
            rewritten = await self._rewrite_query_self_rag(question, current)
            extra = await self.retrieve(question, rewritten, top_k=settings.reranker_top_n)
            if not extra and not current:
                break
            current = self._reciprocal_rank_fusion([current, extra])[:8]

        return current, rounds, reasoning_steps

    async def build_result(
        self,
        question: str,
        intent: QueryIntent,
        contexts: list[RetrievedContext],
        rounds: int,
        reasoning_steps: list[str],
    ) -> QAResult:
        """Generate the grounded answer and assemble the public result."""
        answer_text, reasoning = await self._generate_answer(question, contexts, intent)
        final_reasoning = [*reasoning_steps, *reasoning]
        return QAResult(
            question=question,
            answer=answer_text,
            contexts=contexts,
            intent=intent,
            confidence=self._calc_confidence(contexts),
            reasoning_steps=final_reasoning,
            retrieval_rounds=rounds,
        )

    async def retrieve(
        self,
        question: str,
        rewritten: dict | None = None,
        top_k: int = 8,
    ) -> list[RetrievedContext]:
        """混合检索入口: 向量 + 图谱 → RRF 融合 → BGE 精排"""
        rewritten = rewritten or await self._rewrite_query(question)

        # 向量与图谱分支相互独立，并行执行以降低端到端延迟。
        vector_contexts, graph_contexts = await asyncio.gather(
            self._vector_retrieve(rewritten),
            self._graph_retrieve(question, rewritten),
        )

        merged = self._reciprocal_rank_fusion([vector_contexts, graph_contexts])

        # BGE Reranker 精排: 粗召回 → 精排
        if self.reranker is not None:
            candidates = merged[: settings.reranker_top_n]
            if len(candidates) > 1:
                scores = await self.reranker.arerank(question, [c.content for c in candidates])
                if scores is not None:
                    for ctx, s in zip(candidates, scores):
                        ctx.score = s
                    candidates.sort(key=lambda c: c.score, reverse=True)
                    merged = candidates + merged[settings.reranker_top_n:]
                    merged = self._dedup(merged)

        return merged[:top_k]

    # ── intent classification ────────────────────────────────

    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        try:
            output = await self.structured_llm.invoke(IntentOutput, messages, purpose="intent_classification")
            return QueryIntent(output.intent)
        except Exception:
            pass
        # 降级: 字符串解析
        resp = await self.llm.ainvoke(messages)
        raw = resp.content.strip().lower()
        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    # ── query rewriting ──────────────────────────────────────

    async def _rewrite_query(self, question: str) -> dict:
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        try:
            output = await self.structured_llm.invoke(QueryRewriteOutput, messages, purpose="query_rewrite")
            return {"queries": output.queries, "entities": output.entities, "keywords": output.keywords}
        except Exception:
            pass
        return await self._rewrite_query_fallback(question)

    async def _rewrite_query_fallback(self, question: str) -> dict:
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT + '\n返回 JSON: {"queries": [], "entities": [], "keywords": []}'),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return {"queries": [question], "entities": [], "keywords": []}

    async def _rewrite_query_self_rag(self, question: str, contexts: list[RetrievedContext]) -> dict:
        """Self-RAG 查询改写: 基于已检索内容补充缺失信息点"""
        context_preview = "\n".join(c.content[:120] for c in contexts[:5])
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT + "\n注意: 之前的检索结果不够充分, 请从不同角度改写查询, 覆盖缺失信息点。"),
            HumanMessage(content=f"问题: {question}\n\n已有检索结果(不充分):\n{context_preview}"),
        ]
        try:
            output = await self.structured_llm.invoke(QueryRewriteOutput, messages, purpose="self_rag_rewrite")
            return {"queries": output.queries, "entities": output.entities, "keywords": output.keywords}
        except Exception:
            return await self._rewrite_query_fallback(question)

    async def _evaluate_relevance(self, question: str, contexts: list[RetrievedContext]) -> RelevanceScore | None:
        """Self-RAG: 评估检索结果相关性"""
        context_text = "\n\n".join(f"[{i+1}] {c.content[:200]}" for i, c in enumerate(contexts[:5]))
        messages = [
            SystemMessage(content=RELEVANCE_PROMPT),
            HumanMessage(content=f"问题: {question}\n\n检索结果:\n{context_text}"),
        ]
        try:
            return await self.structured_llm.invoke(RelevanceScore, messages, purpose="relevance_evaluation")
        except Exception as e:
            logger.warning("relevance evaluation failed: %s", e)
            return None

    # ── vector retrieval ─────────────────────────────────────

    async def _vector_retrieve(self, rewritten: dict) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        async def search_one(query: str) -> list[RetrievedContext]:
            try:
                results = await self.vector_store.search(query, top_k=5)
            except Exception:
                return []
            contexts = []
            for doc, score in results:
                metadata = dict(doc.get("metadata", {}))
                metadata["citations"] = self._merge_citations(
                    metadata.get("citations", []),
                    [{
                        "source": doc.get("source", "vector_store"),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "page": metadata.get("page", 0),
                        "heading_path": metadata.get("heading_path", ""),
                    }],
                )
                contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=metadata,
                ))
            return contexts

        queries = [str(q).strip() for q in rewritten.get("queries", []) if str(q).strip()][:3]
        batches = await asyncio.gather(*(search_one(query) for query in queries)) if queries else []
        return [context for batch in batches for context in batch]

    # ── graph retrieval ──────────────────────────────────────

    async def _graph_retrieve(self, _question: str, rewritten: dict) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        # 不执行 LLM 自由生成的 Cypher。实体名始终作为参数传入预定义只读查询，
        # 避免提示注入将检索升级为图数据库写操作。
        entities = [str(e).strip() for e in rewritten.get("entities", []) if str(e).strip()][:5]
        contexts: list[RetrievedContext] = []

        async def neighbors_for(entity: str) -> tuple[str, list[dict]]:
            try:
                return entity, await self.knowledge_graph.get_neighbors(entity, hops=2)
            except Exception:
                return entity, []

        neighbor_batches = await asyncio.gather(*(neighbors_for(entity) for entity in entities)) if entities else []
        for entity, records in neighbor_batches:
            for record in records:
                content = (
                    f"{record.get('source', entity)} "
                    f"--[{', '.join(record.get('relations', []))}]--> "
                    f"{record.get('target', '')} ({record.get('target_type', '')}): "
                    f"{record.get('target_desc', '')}"
                )
                citations = self._citations_from_provenance(record.get("provenance", []))
                contexts.append(RetrievedContext(
                    content=content,
                    source=citations[0]["source"] if citations else "knowledge_graph",
                    score=1.0,
                    retrieval_type="graph",
                    metadata={"entity": entity, "strategy": "neighbors", "citations": citations},
                ))

        path_pairs = [(entities[i], entities[j]) for i in range(len(entities))
                      for j in range(i + 1, min(i + 3, len(entities)))]

        async def paths_for(pair: tuple[str, str]) -> tuple[tuple[str, str], list[dict]]:
            try:
                return pair, await self.knowledge_graph.find_paths(pair[0], pair[1], max_hops=5)
            except Exception:
                return pair, []

        path_batches = await asyncio.gather(*(paths_for(pair) for pair in path_pairs)) if path_pairs else []
        for pair, records in path_batches:
            for record in records:
                nodes = record.get("node_names", [])
                relations = record.get("rel_types", [])
                segments: list[str] = []
                for idx, node in enumerate(nodes):
                    segments.append(str(node))
                    if idx < len(relations):
                        segments.append(f"--[{relations[idx]}]-->")
                citations = self._citations_from_provenance(record.get("provenance", []))
                contexts.append(RetrievedContext(
                    content="推理路径: " + " ".join(segments),
                    source=citations[0]["source"] if citations else "knowledge_graph",
                    score=1.0,
                    retrieval_type="graph",
                    metadata={
                        "from": pair[0], "to": pair[1], "strategy": "shortest_path",
                        "citations": citations,
                    },
                ))
        return contexts

    # ── hybrid reranking ─────────────────────────────────────

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Compatibility wrapper using rank fusion instead of incomparable raw scores."""
        groups = [
            [context for context in contexts if context.retrieval_type == retrieval_type]
            for retrieval_type in ("vector", "graph", "hybrid")
        ]
        return QAAgent._reciprocal_rank_fusion([group for group in groups if group])

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[RetrievedContext]],
        *,
        rank_constant: int = 60,
    ) -> list[RetrievedContext]:
        """Fuse independent ranked lists without comparing their raw score scales."""
        non_empty = [items for items in ranked_lists if items]
        if not non_empty:
            return []
        fused: dict[str, RetrievedContext] = {}
        scores: dict[str, float] = {}
        order: dict[str, int] = {}
        for items in non_empty:
            for rank, context in enumerate(QAAgent._dedup(items), start=1):
                key = QAAgent._context_key(context)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
                if key not in fused:
                    fused[key] = RetrievedContext.from_dict(context.as_dict())
                    order[key] = len(order)
                else:
                    existing = fused[key]
                    existing.metadata["citations"] = QAAgent._merge_citations(
                        existing.metadata.get("citations", []),
                        context.metadata.get("citations", []),
                    )
                    if existing.retrieval_type != context.retrieval_type:
                        existing.retrieval_type = "hybrid"
        max_score = len(non_empty) / (rank_constant + 1)
        for key, context in fused.items():
            context.metadata.setdefault("original_score", context.score)
            context.score = min(scores[key] / max_score, 1.0)
        return sorted(fused.values(), key=lambda item: (-item.score, order[QAAgent._context_key(item)]))

    @staticmethod
    def _dedup(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        unique: dict[str, RetrievedContext] = {}
        for ctx in contexts:
            key = QAAgent._context_key(ctx)
            if key not in unique:
                unique[key] = RetrievedContext.from_dict(ctx.as_dict())
                continue
            existing = unique[key]
            existing.metadata["citations"] = QAAgent._merge_citations(
                existing.metadata.get("citations", []), ctx.metadata.get("citations", []),
            )
            if existing.retrieval_type != ctx.retrieval_type:
                existing.retrieval_type = "hybrid"
            existing.score = max(existing.score, ctx.score)
        return list(unique.values())

    @staticmethod
    def _context_key(context: RetrievedContext) -> str:
        chunk_id = str(context.metadata.get("chunk_id", "")).strip()
        if chunk_id:
            return f"chunk:{chunk_id}"
        normalized = " ".join(context.content.split()).casefold()
        return "content:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _merge_citations(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for citation in (item for group in groups for item in group):
            clean = {
                "source": str(citation.get("source", "")),
                "chunk_id": str(citation.get("chunk_id", "")),
                "page": citation.get("page", 0),
                "heading_path": str(citation.get("heading_path", "")),
            }
            key = (clean["source"], clean["chunk_id"], str(clean["page"]))
            if key not in seen and any(clean.values()):
                seen.add(key)
                merged.append(clean)
        return merged

    @staticmethod
    def _citations_from_provenance(values: list[str]) -> list[dict[str, Any]]:
        citations = []
        for value in values or []:
            source, separator, chunk_id = str(value).rpartition("::")
            citations.append({"source": source if separator else str(value), "chunk_id": chunk_id if separator else ""})
        return QAAgent._merge_citations(citations)

    # ── answer generation ────────────────────────────────────

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        evidence_blocks = []
        for index, context in enumerate(contexts, start=1):
            metadata = context.metadata
            location = " | ".join(
                item for item in (
                    f"chunk: {metadata.get('chunk_id')}" if metadata.get("chunk_id") else "",
                    f"page: {metadata.get('page')}" if metadata.get("page") else "",
                    f"heading: {metadata.get('heading_path')}" if metadata.get("heading_path") else "",
                )
                if item
            )
            location_suffix = f" | {location}" if location else ""
            evidence_blocks.append(
                f"[{index}] 来源: {context.source} | 类型: {context.retrieval_type}"
                f"{location_suffix}\n{context.content}"
            )
        context_text = "\n\n".join(evidence_blocks)
        reasoning_steps = [
            f"包含向量证据: {sum(1 for c in contexts if c.retrieval_type in {'vector', 'hybrid'})} 条",
            f"包含图谱证据: {sum(1 for c in contexts if c.retrieval_type in {'graph', 'hybrid'})} 条",
        ]
        if self.reranker is not None and self.reranker.available:
            reasoning_steps.append("BGE-Reranker 精排完成")

        messages = [
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(content=f"上下文信息:\n{context_text}\n\n用户问题: {question}"),
        ]
        # tags=["final_answer"]: SSE 流式端点据此过滤最终答案的 token
        resp = await self.llm.ainvoke(messages, config={"tags": ["final_answer"]})
        reasoning_steps.append("答案生成完成")
        return resp.content, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        """Return a bounded retrieval-support score, not an answer probability."""
        if not contexts:
            return 0.0
        top_scores = sorted((max(0.0, min(c.score, 1.0)) for c in contexts), reverse=True)[:3]
        return round(sum(top_scores) / len(top_scores), 4)
