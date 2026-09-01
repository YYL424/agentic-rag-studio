"""Opt-in integration test against real Qdrant and Neo4j services."""

from __future__ import annotations

import os
import uuid

import pytest

from agents.knowledge_extract_agent import Entity, Relation
from schema import DocType, DocumentChunk
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCKER_E2E") != "1",
    reason="set RUN_DOCKER_E2E=1 to test real Qdrant and Neo4j services",
)


class DeterministicEmbeddings:
    """Network-free 384-dimensional embeddings for storage contract tests."""

    @staticmethod
    def _vector() -> list[float]:
        return [1.0, *([0.0] * 383)]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector() for _ in texts]


@pytest.mark.asyncio
async def test_real_qdrant_and_neo4j_roundtrip():
    """Write, query, and clean one uniquely named document across both stores."""
    run_id = uuid.uuid4().hex
    source = f"e2e://{run_id}"
    doc_id = f"e2e-{run_id}"
    chunk = DocumentChunk(
        content=f"AgentHub integration probe {run_id}",
        doc_id=doc_id,
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={"source": source},
    )
    head = Entity(name=f"AgentHub-{run_id}", type="System")
    tail = Entity(name=f"Qdrant-{run_id}", type="Database")
    relation = Relation(head=head.name, relation="depends_on", tail=tail.name, confidence=1.0)

    vector_store = VectorStoreService(embeddings=DeterministicEmbeddings())
    knowledge_graph = KnowledgeGraphService()
    await vector_store.init()
    await knowledge_graph.init()
    try:
        assert await vector_store.add_chunks([chunk]) == 1
        matches = await vector_store.search(run_id, top_k=10)
        assert any(item[0]["metadata"]["doc_id"] == doc_id for item in matches)

        await knowledge_graph.upsert_entity(head, source=source, source_chunk_id=chunk.chunk_id)
        await knowledge_graph.upsert_entity(tail, source=source, source_chunk_id=chunk.chunk_id)
        await knowledge_graph.add_relation(relation, source=source, source_chunk_id=chunk.chunk_id)
        rows = await knowledge_graph.execute_readonly_cypher(
            "MATCH (a:Entity {name: $head})-[r:DEPENDS_ON]->(b:Entity {name: $tail}) "
            "RETURN count(r) AS count",
            {"head": head.name, "tail": tail.name},
        )
        assert rows == [{"count": 1}]

        assert await vector_store.delete_by_doc_id(doc_id) == 1
        assert await knowledge_graph.delete_by_source(source) == 2
        assert await knowledge_graph.get_entity(head.name) is None
    finally:
        await vector_store.delete_by_doc_id(doc_id)
        await knowledge_graph.delete_by_source(source)
        await vector_store.close()
        await knowledge_graph.close()
