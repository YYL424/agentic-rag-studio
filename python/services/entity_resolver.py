"""
实体消解服务 (Entity Resolution) — 合并同一实体的不同表述

示例: "腾讯" / "Tencent" / "深圳市腾讯计算机系统有限公司" → 同一实体

两级策略:
  1. 归一化匹配 (快速): 小写/去空白/去后缀 (公司/有限公司/Inc/Ltd)
  2. Embedding 相似度 (精准): bge 模型编码实体名, 余弦相似度 ≥ 阈值则合并

无 embedding 模型时自动降级为纯归一化匹配
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.knowledge_extract_agent import Entity, ExtractionResult
from config import settings

logger = logging.getLogger(__name__)

# 常见组织后缀, 归一化时剔除 ("腾讯科技有限公司" → "腾讯科技")
SUFFIX_PATTERNS = re.compile(
    r"(股份有限公司|有限责任公司|有限公司|集团|公司|股份|Corporation|Corp\.?|Incorporated|Inc\.?|Ltd\.?|LLC|Co\.?|S\.A\.|GmbH|AG)\s*$",
    re.IGNORECASE,
)

SIMILARITY_THRESHOLD = 0.85


class EntityResolver:
    """Embedding 相似度实体消解"""

    def __init__(self, model_name: str | None = None) -> None:
        self._model: Any = None
        self.model_name = model_name or settings.local_embedding_model
        self.merge_stats = {"merged_pairs": 0, "canonical_names": {}}

    # ── public API ───────────────────────────────────────────

    async def resolve(self, results: list[ExtractionResult]) -> list[ExtractionResult]:
        """合并跨 chunk 的同指实体, 返回规范化后的抽取结果"""
        canonical: dict[str, str] = {}   # 归一化键 → 规范名
        name_to_canonical: dict[str, str] = {}  # 实体名 → 规范名

        for result in results:
            new_entities: list[Entity] = []
            for ent in result.entities:
                canon = await self._find_canonical(ent.name, canonical, name_to_canonical)
                if canon != ent.name:
                    self.merge_stats["merged_pairs"] += 1
                    # 合并为规范实体: 保留更长的描述信息
                    merged = self._merge_entity(ent, canon)
                    new_entities.append(merged)
                else:
                    new_entities.append(ent)
            result.entities = new_entities

            # 关系中的头尾实体同步替换为规范名
            for rel in result.relations:
                rel.head = name_to_canonical.get(rel.head, rel.head)
                rel.tail = name_to_canonical.get(rel.tail, rel.tail)

        self.merge_stats["canonical_names"] = canonical
        return results

    async def _find_canonical(
        self,
        name: str,
        canonical: dict[str, str],
        name_to_canonical: dict[str, str],
    ) -> str:
        norm_key = self._normalize(name)
        if not norm_key:
            return name

        # 1. 归一化精确匹配
        if norm_key in canonical:
            name_to_canonical[name] = canonical[norm_key]
            return canonical[norm_key]

        # 2. Embedding 相似度匹配
        similar = await self._find_similar(name, canonical)
        if similar:
            canonical[norm_key] = similar
            name_to_canonical[name] = similar
            return similar

        # 3. 新实体 → 成为规范名
        canonical[norm_key] = name
        name_to_canonical[name] = name
        return name

    async def _find_similar(self, name: str, canonical: dict[str, str]) -> str | None:
        """在已有实体中寻找 embedding 相似度 ≥ 阈值的同指实体"""
        candidates = list(set(canonical.values()))
        # 第二个实体就应该能与首个候选比较；旧条件 <2 会漏掉第一对别名。
        if not candidates:
            return None
        try:
            model = self._get_model()
            if model is None:
                return None
            name_vec = model.encode([name], normalize_embeddings=True)
            cand_vecs = model.encode(candidates, normalize_embeddings=True)
            import numpy as np
            sims = np.dot(cand_vecs, name_vec[0])
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= SIMILARITY_THRESHOLD:
                logger.info("entity resolved: %s → %s (sim=%.3f)", name, candidates[best_idx], sims[best_idx])
                return candidates[best_idx]
        except Exception as e:
            logger.warning("embedding entity resolution unavailable: %s", e)
        return None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("sentence-transformers unavailable, using normalized matching only: %s", e)
                self._model = False
        return self._model if self._model is not False else None

    @staticmethod
    def _normalize(name: str) -> str:
        """归一化: 小写 + 去空白 + 去组织后缀"""
        norm = name.strip().lower()
        norm = re.sub(r"\s+", "", norm)
        norm = SUFFIX_PATTERNS.sub("", norm)
        return norm

    @staticmethod
    def _merge_entity(ent: Entity, canonical_name: str) -> Entity:
        """合并到规范实体: 保留规范名, 信息叠加"""
        merged = Entity(
            name=canonical_name,
            type=ent.type,
            description=ent.description,
        )
        # 别名记录到 properties, 供图谱 alias 查询
        if ent.name != canonical_name:
            aliases = set(merged.properties.get("aliases", []))
            aliases.add(ent.name)
            merged.properties["aliases"] = sorted(aliases)
        return merged
