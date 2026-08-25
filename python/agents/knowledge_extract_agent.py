"""
知识抽取 Agent — 从文档块中提取实体、关系、事件，构建知识图谱三元组

核心能力:
  1. 命名实体识别 (NER)
  2. 关系抽取 (RE)
  3. 事件抽取
  4. 结构化输出 (Pydantic Schema 约束, 替代脆弱的手动 JSON parse)
  5. 实体消解 (Embedding 相似度合并同指实体)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from schema import DocumentChunk

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的：
1. **实体 (entities)**：人名、组织、地点、产品、技术、概念等
2. **关系 (relations)**：实体之间的关系，用三元组 (头实体, 关系, 尾实体) 表示
3. **事件 (events)**：文本中提到的事件，包含触发词和参与者

规则:
- 实体类型: Person, Organization, Location, Product, Technology, Concept, Event, Time
- 关系类型: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- confidence 为 0-1 之间的浮点数
- 只抽取文本中明确出现的信息，不要臆造

示例 1:
文本: "张三于 2020 年加入腾讯，担任微信事业部负责人。"
输出:
  entities: 张三(Person), 腾讯(Organization), 微信(Product)
  relations: (张三, works_at, 腾讯), (张三, 负责, 微信)
  events: 加入(入职事件, 参与者: 张三)

示例 2:
文本: "LangGraph 是 LangChain 团队开发的 Agent 编排框架。"
输出:
  entities: LangGraph(Technology), LangChain(Organization)
  relations: (LangGraph, developed_by, LangChain)
  events: (无)
"""


# ── Pydantic Schema (结构化输出契约) ────────────────────────

class EntityModel(BaseModel):
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型，如 Person/Organization/Product/Technology")
    description: str = Field(default="", description="实体简短描述")


class RelationModel(BaseModel):
    head: str = Field(description="头实体名称")
    relation: str = Field(description="关系类型")
    tail: str = Field(description="尾实体名称")
    confidence: float = Field(default=0.5, ge=0, le=1, description="置信度 0-1")


class EventModel(BaseModel):
    trigger: str = Field(description="事件触发词")
    type: str = Field(description="事件类型")
    participants: list[str] = Field(default_factory=list, description="事件参与者")


class ExtractionOutput(BaseModel):
    """知识抽取结构化输出契约"""
    entities: list[EntityModel] = Field(default_factory=list)
    relations: list[RelationModel] = Field(default_factory=list)
    events: list[EventModel] = Field(default_factory=list)


# ── 系统内部 dataclass (与存储层对接) ───────────────────────

@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class KnowledgeExtractAgent:
    """
    知识抽取 Agent

    工作流:
      receive_chunks → structured_extract (Pydantic) → deduplicate → resolve_entities → output_triples
    """

    BATCH_SIZE = 5

    def __init__(self, entity_resolver: Any = None) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        # with_structured_output: LLM 直接返回 Pydantic 对象 (供应商自适应, 见 _structured_extract)
        self.entity_resolver = entity_resolver

    # ── public API ───────────────────────────────────────────

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """从一组文档块中抽取知识"""
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            results.extend(await asyncio.gather(*(self._extract_from_chunk(chunk) for chunk in batch)))
        merged = self._deduplicate(results)
        if self.entity_resolver:
            merged = await self.entity_resolver.resolve(merged)
        return merged

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """从单段文本中抽取知识"""
        return await self._extract_from_text(text, chunk_id)

    # ── core extraction ──────────────────────────────────────

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        result = await self._structured_extract(text)
        if result is not None:
            return self._to_extraction_result(result, source_id)
        # 降级: 供应商不支持结构化输出时回退到 JSON 解析路径
        return await self._json_extract(text, source_id)

    async def _structured_extract(self, text: str) -> ExtractionOutput | None:
        """
        Pydantic 结构化输出 — 主路径 (供应商自适应):
          1. function_calling: OpenAI 等主流供应商
          2. json_mode: DeepSeek 等 (要求 prompt 含 "json" 字样)
          全部失败 → 返回 None, 由 _json_extract 手动解析兜底
        """
        for method, prompt_suffix in (
            ("function_calling", None),
            ("json_mode", "\n\nOutput a json object matching the schema."),
        ):
            try:
                structured = self.llm.with_structured_output(ExtractionOutput, method=method)
                system = EXTRACTION_SYSTEM_PROMPT + (prompt_suffix or "")
                messages = [
                    SystemMessage(content=system),
                    HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
                ]
                return await structured.ainvoke(messages)
            except Exception as e:
                logger.warning("structured output method=%s failed (%s)", method, str(e)[:120])
        return None

    async def _json_extract(self, text: str, source_id: str) -> ExtractionResult:
        """JSON 字符串解析 — 降级路径"""
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT + "\n\n请严格按照 JSON 格式返回，只返回 JSON，不要包含其他文字。"),
            HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            return self._parse_response(resp.content, source_id)
        except Exception:
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

    @staticmethod
    def _to_extraction_result(output: ExtractionOutput, source_id: str) -> ExtractionResult:
        """Pydantic 模型 → 系统 dataclass"""
        return ExtractionResult(
            entities=[
                Entity(name=e.name, type=e.type or "Concept", description=e.description)
                for e in output.entities
                if e.name
            ],
            relations=[
                Relation(head=r.head, relation=r.relation or "related_to", tail=r.tail, confidence=r.confidence)
                for r in output.relations
                if r.head and r.tail
            ],
            events=[
                KnowledgeEvent(trigger=ev.trigger, type=ev.type, participants=ev.participants)
                for ev in output.events
            ],
            source_chunk_id=source_id,
        )

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(
                head=r.get("head", ""),
                relation=r.get("relation", "related_to"),
                tail=r.get("tail", ""),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in data.get("relations", [])
            if r.get("head") and r.get("tail")
        ]
        events = [
            KnowledgeEvent(
                trigger=ev.get("trigger", ""),
                type=ev.get("type", ""),
                participants=ev.get("participants", []),
            )
            for ev in data.get("events", [])
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        )

    # ── deduplication ────────────────────────────────────────

    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        chunk 内去重，保留跨 chunk 的重复出现。

        跨 chunk 不能直接删除重复实体，否则图谱会丢失该实体在其他 chunk
        中的 provenance，删除或更新其中一个 chunk 时可能误删共享事实。
        全局实体合并由 Neo4j MERGE 和 EntityResolver 负责。
        """
        deduped: list[ExtractionResult] = []

        for result in results:
            seen_entities: set[str] = set()
            seen_relations: set[tuple[str, str, str]] = set()
            unique_entities: list[Entity] = []
            for ent in result.entities:
                key = f"{ent.name}::{ent.type}"
                if key not in seen_entities:
                    seen_entities.add(key)
                    unique_entities.append(ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                key = (rel.head, rel.relation, rel.tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(rel)

            deduped.append(ExtractionResult(
                entities=unique_entities,
                relations=unique_relations,
                events=result.events,
                source_chunk_id=result.source_chunk_id,
            ))
        return deduped
