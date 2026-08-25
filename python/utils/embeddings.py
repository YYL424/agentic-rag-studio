"""
Embedding 工厂 — 支持 API (OpenAI 兼容) 和本地 sentence-transformers 两种后端
"""

from __future__ import annotations

import asyncio
from typing import Any


def create_embeddings():
    """根据配置返回 embedding 实例（同步接口，含 aembed_* 异步方法）"""
    from config import settings

    if settings.embedding_backend == "local":
        return _create_local_embeddings()
    else:
        return _create_api_embeddings()


def _create_api_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from config import settings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _create_local_embeddings():
    from config import settings
    return LocalEmbeddings(model_name=settings.local_embedding_model)


class LocalEmbeddings:
    """本地 sentence-transformers 嵌入，接口兼容 OpenAIEmbeddings"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
