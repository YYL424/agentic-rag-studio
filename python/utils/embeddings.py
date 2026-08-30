"""Embedding 工厂 — 支持 API 与本地模型，并为轻量镜像提供 ONNX 回退。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _download_model_file(
    url: str,
    destination: str,
    expected_sha256: str,
    chunk_size: int = 64 * 1024,
    attempts: int = 3,
) -> None:
    """在代理/慢网络下可靠下载模型，并在替换目标文件前校验哈希。"""
    import httpx

    target = Path(destination)
    partial = target.with_name(f"{target.name}.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            digest = hashlib.sha256()
            timeout = httpx.Timeout(connect=60.0, read=180.0, write=60.0, pool=60.0)
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for data in response.iter_bytes(chunk_size=chunk_size):
                        output.write(data)
                        digest.update(data)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("Downloaded model does not match expected SHA256")
            partial.replace(target)
            return
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                logger.warning("ONNX 模型下载失败（%s/%s），准备重试: %s", attempt, attempts, exc)

    raise RuntimeError(f"ONNX 模型下载失败，已重试 {attempts} 次") from last_error


def _create_chroma_onnx_model():
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
        ONNXMiniLM_L6_V2,
    )

    class ResilientONNXMiniLM(ONNXMiniLM_L6_V2):
        def _download(self, url: str, fname: str, chunk_size: int = 1024) -> None:
            _download_model_file(url, fname, self._MODEL_SHA256, chunk_size=max(chunk_size, 64 * 1024))

    return ResilientONNXMiniLM()


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
    """本地嵌入，接口兼容 ``OpenAIEmbeddings``。

    完整环境优先使用配置的 sentence-transformers 模型；轻量环境没有该可选
    依赖时，自动使用 Chroma 已内置的 ONNX MiniLM，避免为了 CPU 容器安装
    PyTorch/CUDA 依赖。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._backend = "chroma-onnx/all-MiniLM-L6-v2"
            self._model = _create_chroma_onnx_model()
            logger.warning(
                "sentence-transformers 未安装，使用轻量 ONNX embedding 回退；"
                "如需模型 %s，请安装 requirements-ml.txt",
                model_name,
            )
        else:
            self._backend = f"sentence-transformers/{model_name}"
            self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend.startswith("sentence-transformers/"):
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        vectors = self._model(texts)
        return [self._as_list(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self._backend.startswith("sentence-transformers/"):
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._as_list(self._model([text])[0])

    @staticmethod
    def _as_list(vector: Any) -> list[float]:
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        return [float(value) for value in values]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
