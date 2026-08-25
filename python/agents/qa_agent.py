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


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)
    retrieval_rounds: int = 1


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
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
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

        # 结构化输出 (with_structured_output), 失败时降级到字符串解析
        self._intent_llm = self.llm.with_structured_output(IntentOutput)
        self._rewrite_llm = self.llm.with_structured_output(QueryRewriteOutput)
        self._relevance_llm = self.llm.with_structured_output(RelevanceScore)

    # ── public API ───────────────────────────────────────────

    async def answer(self, question: str) -> QAResult:
        """完整问答流程 (Self-RAG + Reranker)"""
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        rounds = 1
        all_contexts = await self.retrieve(question, rewritten, top_k=8)
        reasoning_steps = [
            f"识别问题意图: {intent.value}",
            f"第 {rounds} 轮检索到 {len(all_contexts)} 条上下文",
        ]

        # ── Self-RAG 自适应检索循环 ──────────────────────────
        if settings.enable_self_rag and all_contexts:
            relevance = await self._evaluate_relevance(question, all_contexts)
            while (
                relevance is not None
                and relevance.score < settings.self_rag_threshold
                and rounds < settings.self_rag_max_rounds
            ):
                rounds += 1
                reasoning_steps.append(
                    f"检索相关性 {relevance.score:.2f} 低于阈值 {settings.self_rag_threshold}, 改写查询重检 (第 {rounds} 轮)"
                )
                rewritten = await self._rewrite_query_self_rag(question, all_contexts)
                extra = await self.retrieve(question, rewritten, top_k=settings.reranker_top_n)

                # 合并全部候选, 用 BGE-Reranker 对全集重新精排 (而非简单混合截断)
                candidates = self._dedup(all_contexts + extra)
                if self.reranker is not None and len(candidates) > 1:
                    scores = await self.reranker.arerank(question, [c.content for c in candidates])
                    if scores is not None:
                        for ctx, s in zip(candidates, scores):
                            ctx.score = s
                        candidates.sort(key=lambda c: c.score, reverse=True)

                all_contexts = candidates[:8]
                relevance = await self._evaluate_relevance(question, all_contexts)

        answer_text, reasoning = await self._generate_answer(question, all_contexts, intent)
        reasoning_steps.extend(reasoning)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=all_contexts,
            intent=intent,
            confidence=self._calc_confidence(all_contexts),
            reasoning_steps=reasoning_steps,
            retrieval_rounds=rounds,
        )

    async def retrieve(
        self,
        question: str,
        rewritten: dict | None = None,
        top_k: int = 8,
    ) -> list[RetrievedContext]:
        """混合检索入口: 向量 + 图谱 → 加权粗排 → BGE 精排"""
        rewritten = rewritten or await self._rewrite_query(question)

        # 向量与图谱分支相互独立，并行执行以降低端到端延迟。
        vector_contexts, graph_contexts = await asyncio.gather(
            self._vector_retrieve(rewritten),
            self._graph_retrieve(question, rewritten),
        )

        merged = self._hybrid_rerank(vector_contexts + graph_contexts)

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
            output = await self._intent_llm.ainvoke(messages)
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
            output = await self._rewrite_llm.ainvoke(messages)
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
            output = await self._rewrite_llm.ainvoke(messages)
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
            return await self._relevance_llm.ainvoke(messages)
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
            return [
                RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=doc.get("metadata", {}),
                )
                for doc, score in results
            ]

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
                contexts.append(RetrievedContext(
                    content=content,
                    source="knowledge_graph",
                    score=0.8,
                    retrieval_type="graph",
                    metadata={"entity": entity, "strategy": "neighbors"},
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
                contexts.append(RetrievedContext(
                    content="推理路径: " + " ".join(segments),
                    source="knowledge_graph",
                    score=0.9,
                    retrieval_type="graph",
                    metadata={"from": pair[0], "to": pair[1], "strategy": "shortest_path"},
                ))
        return contexts

    # ── hybrid reranking ─────────────────────────────────────

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        混合加权粗排: 向量分数 + 图谱分数加权
        图谱检索结果天然带有结构化关系，给予略高权重
        (精排交给 BGE Reranker)
        """
        weight_map = {"vector": 1.0, "graph": 1.2, "hybrid": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)
        return QAAgent._dedup(sorted(contexts, key=lambda c: c.score, reverse=True))

    @staticmethod
    def _dedup(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)
        return unique

    # ── answer generation ────────────────────────────────────

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        context_text = "\n\n".join(
            f"[来源 {i+1}: {c.source} | 类型: {c.retrieval_type} | 分数: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"向量检索: {sum(1 for c in contexts if c.retrieval_type == 'vector')} 条",
            f"图谱检索: {sum(1 for c in contexts if c.retrieval_type == 'graph')} 条",
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
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
