"""
MCP (Model Context Protocol) Server — 将核心能力封装为 MCP tools

运行方式 (stdio):
  python mcp_server.py

接入客户端 (Claude Desktop / Cline / Cursor 等):
  "mcpServers": {
    "agent-knowledge-hub": {
      "command": "python",
      "args": ["<项目路径>/python/mcp_server.py"]
    }
  }

暴露的工具:
  - parse_document: 解析文档 → 结构化 chunk (marker-pdf/docling/pymupdf 降级链)
  - search_knowledge: 向量 + 图谱混合检索
  - ask: 完整 RAG 问答 (Self-RAG + Reranker)
  - query_graph: 执行 Cypher 查询
  - get_stats: 知识库统计
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _build_mcp():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agent-knowledge-hub")

    # 懒加载服务 (仅当对应工具被调用时初始化)
    _state: dict[str, Any] = {}

    async def _get_parser():
        if "parser" not in _state:
            from agents.doc_parser_agent import DocParserAgent
            _state["parser"] = DocParserAgent()
        return _state["parser"]

    async def _get_qa():
        if "qa" not in _state:
            from services.knowledge_graph import KnowledgeGraphService
            from services.vector_store import VectorStoreService

            vector_store = VectorStoreService()
            try:
                await vector_store.init()
            except Exception:
                vector_store = None
            knowledge_graph = KnowledgeGraphService()
            try:
                await knowledge_graph.init()
            except Exception:
                knowledge_graph = None

            from agents.qa_agent import QAAgent
            _state["qa"] = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
            _state["vector_store"] = vector_store
            _state["knowledge_graph"] = knowledge_graph
        return _state["qa"]

    def _managed_document_path(file_path: str) -> str:
        from config import settings

        root = Path(settings.upload_dir).resolve()
        candidate = Path(file_path)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as e:
            raise ValueError("file_path must be inside configured upload_dir") from e
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return str(resolved)

    @mcp.tool()
    async def parse_document(file_path: str) -> list[dict[str, Any]]:
        """解析文档 (PDF/Word/PPT/图片/表格/Markdown) 并返回结构化 chunk 列表"""
        parser = await _get_parser()
        chunks = await parser.parse(_managed_document_path(file_path))
        return [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "doc_type": c.doc_type.value,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    @mcp.tool()
    async def search_knowledge(question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """从向量库和知识图谱中混合检索相关知识"""
        qa = await _get_qa()
        contexts = await qa.retrieve(question, top_k=max(1, min(top_k, 20)))
        return [
            {
                "content": c.content,
                "source": c.source,
                "score": c.score,
                "retrieval_type": c.retrieval_type,
            }
            for c in contexts
        ]

    @mcp.tool()
    async def ask(question: str) -> dict[str, Any]:
        """完整 RAG 问答: 混合检索 + 重排序 + Self-RAG + 答案生成"""
        qa = await _get_qa()
        result = await qa.answer(question)
        return {
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "intent": result.intent.value,
            "sources": [
                {"content": c.content[:200], "source": c.source, "score": c.score}
                for c in result.contexts
            ],
            "reasoning_steps": result.reasoning_steps,
        }

    @mcp.tool()
    async def query_graph(cypher: str) -> list[dict[str, Any]]:
        """执行 Cypher 查询知识图谱"""
        kg = _state.get("knowledge_graph")
        if kg is None:
            from services.knowledge_graph import KnowledgeGraphService
            kg = KnowledgeGraphService()
            await kg.init()
            _state["knowledge_graph"] = kg
        return await kg.execute_readonly_cypher(cypher)

    @mcp.tool()
    async def get_stats() -> dict[str, Any]:
        """获取知识库统计信息 (向量数 / 实体数 / 关系数)"""
        stats: dict[str, Any] = {}
        vector_store = _state.get("vector_store")
        if vector_store:
            stats["vector_store"] = await vector_store.get_stats()
        kg = _state.get("knowledge_graph")
        if kg:
            stats["knowledge_graph"] = await kg.get_stats()
        return stats

    return mcp


if __name__ == "__main__":
    mcp = _build_mcp()
    # stdio 传输: 供 Claude Desktop / Cline 等 MCP 客户端连接
    mcp.run(transport="stdio")
