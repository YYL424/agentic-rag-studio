"""
重排序服务 (Reranker) — 检索结果精排

BGE-Reranker-v2-m3 交叉编码器, 对 query-document 对打分:
  - 比纯向量余弦相似度更精准 (输入为 query+doc 联合编码, 捕获交互语义)
  - 检索阶段粗召回 (top 20) → 重排阶段精排 (top 8)

模型未安装/不可用时自动降级为调用方原有的加权排序
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """BGE Reranker 封装 — 懒加载, 失败自动降级"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: Any = None

    @property
    def available(self) -> bool:
        """重排模型是否可用 (不可用时调用方走降级排序)"""
        return self._get_model() is not None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info("reranker loaded: %s", self.model_name)
            except Exception as e:
                logger.warning("reranker unavailable (%s), falling back to weighted hybrid sort", e)
                self._model = False
        return self._model if self._model is not False else None

    def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """
        对 query-document 对打分, 返回与 documents 等长的相似度分数列表
        模型不可用时返回 None
        """
        model = self._get_model()
        if model is None:
            return None
        try:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs)
            # 归一化到 0-1 (sigmoid), 与向量分数同一量纲
            import numpy as np
            return (1.0 / (1.0 + np.exp(-np.asarray(scores)))).tolist()
        except Exception as e:
            logger.warning("rerank failed: %s", e)
            return None

    async def arerank(self, query: str, documents: list[str]) -> list[float] | None:
        import asyncio
        return await asyncio.to_thread(self.rerank, query, documents)
