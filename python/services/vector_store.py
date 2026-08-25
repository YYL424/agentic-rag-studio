"""
向量存储服务 — 支持 ChromaDB / PGVector / Qdrant 三后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索 (支持元数据过滤)
  3. 按 doc_id / chunk_id 删除（支持增量更新）

Qdrant 模式:
  - qdrant_url 配置了 → 服务端模式
  - qdrant_url 留空 → 嵌入式本地模式 (path 存储, 无需 Docker, 开发友好)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings
from schema import DocumentChunk
from utils.embeddings import create_embeddings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector / Qdrant"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self, embeddings: Any = None) -> None:
        # 延迟创建 embedding，避免仅导入 API/MCP 模块就加载本地模型。
        # 这也让轻量测试可以在未安装 sentence-transformers 时启动。
        self.embeddings = embeddings
        self._store: Any = None
        self._client: Any = None
        self._backend = settings.vector_store_type
        self._qdrant_dim: int | None = None

    # ── initialization (factory 模式) ────────────────────────

    async def init(self) -> None:
        """根据 settings.vector_store_type 选择后端 (factory)"""
        if self.embeddings is None:
            self.embeddings = create_embeddings()
        self._store = await self._create_backend()

    async def close(self) -> None:
        """释放嵌入式数据库文件句柄。"""
        store, self._store = self._store, None
        client, self._client = self._client, None
        close = getattr(store, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
        # Chroma Collection 不暴露 close；其 client system 持有本地 SQLite 连接。
        system = getattr(client, "_system", None)
        stop = getattr(system, "stop", None)
        if callable(stop):
            await asyncio.to_thread(stop)

    async def _create_backend(self):
        """
        factory: 按配置创建后端存储客户端
          qdrant   → Qdrant (url 服务端 / path 嵌入式双模式)
          pgvector → PGVector (langchain)
          其他     → Chroma (默认, 保证兼容性)
        """
        if settings.vector_store_type == "qdrant":
            return await self._create_qdrant()
        elif settings.vector_store_type == "pgvector":
            return self._create_pgvector()
        return self._create_chroma()

    def _create_chroma(self):
        """Chroma: 本地持久化模式 (chroma_path) 或服务端模式 (HttpClient)"""
        import chromadb
        if settings.chroma_path:
            client = chromadb.PersistentClient(path=settings.chroma_path)
            logger.info("Chroma local mode: %s", settings.chroma_path)
        else:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._client = client
        return client.get_or_create_collection(name=self.COLLECTION_NAME)

    def _create_pgvector(self):
        from langchain_community.vectorstores import PGVector
        return PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    async def _create_qdrant(self):
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            logger.info("Qdrant server mode: %s", settings.qdrant_url)
        else:
            client = QdrantClient(path=settings.qdrant_path)
            logger.info("Qdrant embedded mode: %s", settings.qdrant_path)

        self._store = client
        await self._ensure_qdrant_collection()
        return client

    async def _ensure_qdrant_collection(self) -> None:
        """确保 collection 存在 (懒创建, 维度首次探测)"""
        from qdrant_client.models import Distance, VectorParams

        existing = (await asyncio.to_thread(self._store.get_collections)).collections
        if any(c.name == self.COLLECTION_NAME for c in existing):
            return

        # 探测 embedding 维度
        dim = len(await self.embeddings.aembed_query("dimension probe"))
        self._qdrant_dim = dim
        await asyncio.to_thread(
            self._store.create_collection,
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并存储文档块"""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [self._flatten_metadata(c) for c in chunks]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            await asyncio.to_thread(
                self._store.upsert,
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
        elif self._backend == "qdrant":
            await self._qdrant_add(ids, texts, metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    @staticmethod
    def _flatten_metadata(chunk: DocumentChunk) -> dict[str, Any]:
        """提取关键元数据 (Qdrant payload 支持复杂元数据过滤)"""
        meta = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_type": chunk.doc_type.value,
            "chunk_index": chunk.chunk_index,
            "source": chunk.metadata.get("source", ""),
            "heading_path": chunk.metadata.get("heading_path", ""),
            "is_table": chunk.metadata.get("is_table", False),
            "is_image": chunk.metadata.get("is_image", False),
            "page": chunk.metadata.get("page", 0),
        }
        return meta

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """chunk_id → UUID5 (Qdrant 嵌入式模式只支持 UUID 点 ID)"""
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    async def _qdrant_add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        vectors = await self.embeddings.aembed_documents(texts)
        points = [
            PointStruct(
                id=self._chunk_id_to_uuid(pid),
                vector=vec,
                payload={**meta, "content": text},
            )
            for pid, vec, text, meta in zip(ids, vectors, texts, metadatas)
        ]
        await asyncio.to_thread(
            self._store.upsert,
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表"""
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            results = await asyncio.to_thread(
                self._store.query,
                query_embeddings=[q_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out: list[tuple[dict, float]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                score = 1.0 - dist  # cosine distance → similarity
                out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
            return out
        elif self._backend == "qdrant":
            return await self._qdrant_search(query, top_k)
        else:
            results = await self._store.asimilarity_search_with_score(query, k=top_k)
            return [
                ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
                for doc, score in results
            ]

    async def _qdrant_search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        q_vec = await self.embeddings.aembed_query(query)
        results = await asyncio.to_thread(
            self._store.query_points,
            collection_name=self.COLLECTION_NAME,
            query=q_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        out: list[tuple[dict, float]] = []
        for point in results.points:
            payload = point.payload or {}
            out.append((
                {
                    "content": payload.get("content", ""),
                    "source": payload.get("source", ""),
                    "metadata": payload,
                },
                point.score or 0.0,
            ))
        return out

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if self._backend == "chroma":
            existing = await asyncio.to_thread(self._store.get, where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                await asyncio.to_thread(self._store.delete, ids=ids)
            return len(ids)
        elif self._backend == "qdrant":
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            points, _ = await asyncio.to_thread(
                self._store.scroll,
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=10_000,
                with_vectors=False,
            )
            ids = [p.id for p in points]
            if ids:
                await asyncio.to_thread(
                    self._store.delete,
                    collection_name=self.COLLECTION_NAME,
                    points_selector=ids,
                )
            return len(ids)
        return 0

    async def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """按 chunk_id 列表精确删除 (增量更新: 只删除变化的 chunk)"""
        if not chunk_ids:
            return 0
        if self._backend == "chroma":
            await asyncio.to_thread(self._store.delete, ids=chunk_ids)
            return len(chunk_ids)
        elif self._backend == "qdrant":
            uuid_ids = [self._chunk_id_to_uuid(cid) for cid in chunk_ids]
            await asyncio.to_thread(
                self._store.delete,
                collection_name=self.COLLECTION_NAME,
                points_selector=uuid_ids,
            )
            return len(chunk_ids)
        return 0

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if self._backend == "chroma":
            if self._store is not None:
                count = await asyncio.to_thread(self._store.count)
                return {"backend": "chroma", "total_vectors": count, "collection": self.COLLECTION_NAME}
            return {"backend": "chroma", "total_vectors": 0, "collection": self.COLLECTION_NAME, "status": "not_initialized"}
        elif self._backend == "qdrant":
            try:
                info = await asyncio.to_thread(self._store.get_collection, self.COLLECTION_NAME)
                return {"backend": "qdrant", "total_vectors": info.points_count or 0, "collection": self.COLLECTION_NAME,
                        "mode": "server" if settings.qdrant_url else "embedded"}
            except Exception:
                return {"backend": "qdrant", "total_vectors": 0, "collection": self.COLLECTION_NAME, "status": "not_initialized"}
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
