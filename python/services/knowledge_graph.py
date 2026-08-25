"""
知识图谱服务 — Neo4j 图数据库操作

职责:
  1. 实体 (Node) CRUD — 带版本号和时间戳
  2. 关系 (Relationship) CRUD
  3. Cypher 查询执行
  4. 子图检索（多跳遍历）
  5. 按来源删除（支持增量更新）
"""

from __future__ import annotations

import re
import time
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings


class KnowledgeGraphService:
    """Neo4j 知识图谱服务"""

    ALLOWED_RELATION_TYPES = {
        "BELONGS_TO",
        "WORKS_AT",
        "LOCATED_IN",
        "DEVELOPED_BY",
        "RELATED_TO",
        "PART_OF",
        "USES",
        "DEPENDS_ON",
    }
    _WRITE_CYPHER = re.compile(
        r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|CALL|LOAD\s+CSV|"
        r"FOREACH|GRANT|DENY|REVOKE|TERMINATE|START|STOP)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._driver: Any = None

    # ── lifecycle ────────────────────────────────────────────

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        """创建常用索引以加速查询"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                await session.run(q)

    # ── entity operations ────────────────────────────────────

    async def upsert_entity(
        self,
        entity: Entity,
        version: int = 1,
        source: str = "",
        source_chunk_id: str = "",
    ) -> None:
        """
        创建或更新实体节点 — MERGE 语义
        带版本号和时间戳，支持增量更新追踪
        """
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.version = $version,
            e.provenance = [],
            e.aliases = $aliases,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.aliases = reduce(acc = coalesce(e.aliases, []), a IN $aliases |
                CASE WHEN a IN acc THEN acc ELSE acc + a END),
            e.updated_at = $now
        SET e.provenance = CASE
            WHEN $provenance = '' OR $provenance IN coalesce(e.provenance, [])
                THEN coalesce(e.provenance, [])
            ELSE coalesce(e.provenance, []) + $provenance
        END
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "version": version,
                "provenance": self._provenance_key(source, source_chunk_id),
                "aliases": entity.properties.get("aliases", []),
                "now": int(time.time()),
            })

    async def add_relation(
        self,
        relation: Relation,
        source: str = "",
        source_chunk_id: str = "",
    ) -> None:
        """创建实体间关系"""
        rel_type = self._normalize_relation_type(relation.relation)
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_type}]->(t)
        ON CREATE SET r.provenance = []
        SET r.confidence = $confidence,
            r.updated_at = $now,
            r.provenance = CASE
                WHEN $provenance = '' OR $provenance IN coalesce(r.provenance, [])
                    THEN coalesce(r.provenance, [])
                ELSE coalesce(r.provenance, []) + $provenance
            END
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head,
                "tail": relation.tail,
                "confidence": relation.confidence,
                "provenance": self._provenance_key(source, source_chunk_id),
                "now": int(time.time()),
            })

    @classmethod
    def _normalize_relation_type(cls, relation: str) -> str:
        """仅允许预定义关系类型，LLM 的任意输出降级为 RELATED_TO。"""
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", relation.strip()).strip("_").upper()
        return normalized if normalized in cls.ALLOWED_RELATION_TYPES else "RELATED_TO"

    @staticmethod
    def _provenance_key(source: str, source_chunk_id: str) -> str:
        if source and source_chunk_id:
            return f"{source}::{source_chunk_id}"
        return source or source_chunk_id

    # ── query operations ─────────────────────────────────────

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """执行任意 Cypher 查询"""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    @classmethod
    def validate_readonly_cypher(cls, cypher: str) -> str:
        """校验外部/LLM Cypher，只允许单条只读查询。"""
        query = cypher.strip()
        if not query or ";" in query.rstrip(";"):
            raise ValueError("只允许执行一条 Cypher 查询")
        if cls._WRITE_CYPHER.search(query):
            raise ValueError("只读接口禁止写入型 Cypher")
        if not re.match(r"^(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b", query, re.IGNORECASE):
            raise ValueError("只读 Cypher 必须以 MATCH/WITH/UNWIND/RETURN 开始")
        return query.rstrip(";").strip()

    async def execute_readonly_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        return await self.execute_cypher(self.validate_readonly_cypher(cypher), params)

    async def get_entity(self, name: str) -> dict | None:
        """查询单个实体"""
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """
        多跳子图检索 — GraphRAG 的核心能力
        从指定实体出发，遍历 N 跳内的所有关联实体和关系
        """
        hops = max(1, min(int(hops), 5))
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
        RETURN
            start.name AS source,
            [r IN relationships(path) | type(r)] AS relations,
            neighbor.name AS target,
            neighbor.type AS target_type,
            neighbor.description AS target_desc
        LIMIT 50
        """
        return await self.execute_cypher(cypher, {"name": entity_name})

    async def find_paths(self, name_a: str, name_b: str, max_hops: int = 5) -> list[dict]:
        """使用参数化实体名查找最短路径；跳数被限制在 1-5。"""
        max_hops = max(1, min(int(max_hops), 5))
        cypher = f"""
        MATCH path = shortestPath(
            (a:Entity {{name: $name_a}})-[*..{max_hops}]-(b:Entity {{name: $name_b}})
        )
        RETURN
            [n IN nodes(path) | n.name] AS node_names,
            [r IN relationships(path) | type(r)] AS rel_types
        LIMIT 3
        """
        return await self.execute_cypher(cypher, {"name_a": name_a, "name_b": name_b})

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        """模糊搜索实体"""
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    # ── delete operations ────────────────────────────────────

    async def delete_by_source(self, source: str) -> int:
        """删除某文档的 provenance；仍被其他文档引用的实体会保留。"""
        prefix = f"{source}::"
        await self.execute_cypher(
            """
            MATCH ()-[r]->()
            WHERE any(p IN coalesce(r.provenance, []) WHERE p = $source OR p STARTS WITH $prefix)
            SET r.provenance = [p IN coalesce(r.provenance, [])
                                WHERE NOT (p = $source OR p STARTS WITH $prefix)]
            WITH r WHERE size(r.provenance) = 0
            DELETE r
            """,
            {"source": source, "prefix": prefix},
        )
        records = await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE any(p IN coalesce(e.provenance, []) WHERE p = $source OR p STARTS WITH $prefix)
            SET e.provenance = [p IN coalesce(e.provenance, [])
                                WHERE NOT (p = $source OR p STARTS WITH $prefix)]
            WITH e WHERE size(e.provenance) = 0
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            {"source": source, "prefix": prefix},
        )
        return records[0].get("deleted", 0) if records else 0

    async def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """移除变化 chunk 的图谱 provenance，并清理失去全部来源的事实。"""
        if not chunk_ids:
            return 0
        await self.execute_cypher(
            """
            MATCH ()-[r]->()
            WHERE any(p IN coalesce(r.provenance, [])
                      WHERE any(cid IN $chunk_ids WHERE p = cid OR p ENDS WITH '::' + cid))
            SET r.provenance = [p IN coalesce(r.provenance, [])
                                WHERE none(cid IN $chunk_ids WHERE p = cid OR p ENDS WITH '::' + cid)]
            WITH r WHERE size(r.provenance) = 0
            DELETE r
            """,
            {"chunk_ids": chunk_ids},
        )
        records = await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE any(p IN coalesce(e.provenance, [])
                      WHERE any(cid IN $chunk_ids WHERE p = cid OR p ENDS WITH '::' + cid))
            SET e.provenance = [p IN coalesce(e.provenance, [])
                                WHERE none(cid IN $chunk_ids WHERE p = cid OR p ENDS WITH '::' + cid)]
            WITH e WHERE size(e.provenance) = 0
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            {"chunk_ids": chunk_ids},
        )
        return records[0].get("deleted", 0) if records else 0

    # ── stats ────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """获取图谱统计信息"""
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }
