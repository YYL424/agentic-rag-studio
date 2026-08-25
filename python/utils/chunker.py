"""
语义分块器 — 三级降级策略

  1. 语义分块 (内置实现): embedding 检测相邻句子的语义边界
  2. 结构分块 (结构感知): 按标题层级 + 段落边界切分, 保留 heading_path
  3. 固定分块 (fallback): 固定字符数 + 重叠 (旧行为, 保证任何环境下可用)

所有 chunk 均携带结构化元数据:
  - heading_path: 标题路径 (如 "## 3.2 架构设计 / ### 3.2.1 为什么选 LangGraph")
  - is_table / is_image: 表格/图片标记
  - page: 页码
"""

from __future__ import annotations

import logging
from typing import Any

from schema import DocType, DocumentChunk
from services.document_parser import ParseBlock

logger = logging.getLogger(__name__)


class DocumentChunker:
    """将 ParseBlock 列表切分为 DocumentChunk 列表"""

    def __init__(
        self,
        embeddings: Any = None,
        semantic_enabled: bool = True,
        breakpoint_threshold: int = 85,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.embeddings = embeddings
        self.semantic_enabled = semantic_enabled
        self.breakpoint_threshold = breakpoint_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── public API ───────────────────────────────────────────

    def chunk(
        self,
        blocks: list[ParseBlock],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        """切分解析块为文档块, 保留结构化元数据"""
        merged = self._merge_small_blocks(blocks)

        if self.semantic_enabled:
            semantic = self._semantic_chunk(merged)
            if semantic:
                return self._to_document_chunks(semantic, doc_id, doc_type, source)

        structural = self._structural_chunk(merged)
        return self._to_document_chunks(structural, doc_id, doc_type, source)

    # ── Step 0: 小段落合并 ───────────────────────────────────

    @staticmethod
    def _merge_small_blocks(blocks: list[ParseBlock], min_len: int = 60) -> list[ParseBlock]:
        """将过短的相邻段落合并, 避免 chunk 碎片化"""
        merged: list[ParseBlock] = []
        for block in blocks:
            if block.kind == "heading" or block.is_table:
                merged.append(block)
                continue
            if merged and merged[-1].kind == "paragraph" and len(merged[-1].content) < min_len:
                merged[-1].content += "\n" + block.content
            else:
                merged.append(block)
        return merged

    # ── Step 1: 语义分块 ─────────────────────────────────────

    def _semantic_chunk(self, blocks: list[ParseBlock]) -> list[ParseBlock] | None:
        """
        基于相邻句向量余弦距离的语义分块。

        项目内置该实现，避免依赖已停止维护的 langchain-experimental；
        同时施加 chunk_size 硬上限，防止纯语义边界产生超长 chunk。
        """
        if not self.embeddings:
            return None
        try:
            result: list[ParseBlock] = []
            for block in blocks:
                if block.is_table:
                    result.append(block)
                    continue

                pieces = self._semantic_split_text(block.content)
                for piece in pieces:
                    result.append(ParseBlock(
                        kind=block.kind,
                        content=piece,
                        page=block.page,
                        heading_path=list(block.heading_path),
                        is_table=False,
                        is_image=block.is_image,
                        ocr_text=block.ocr_text,
                    ))
            return result if any(len(b.content) > 20 for b in result) else None
        except Exception as e:
            logger.warning("semantic chunking unavailable, falling back to structural chunking: %s", e)
            return None

    def _semantic_split_text(self, text: str) -> list[str]:
        """把文本按语义断点合并为带长度上限的片段。"""
        import math
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if s.strip()]
        if len(sentences) < 2:
            return self._split_long_paragraph(text) if len(text) > self.chunk_size else [text]

        vectors = self.embeddings.embed_documents(sentences)

        def cosine_distance(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if not norm_a or not norm_b:
                return 0.0
            return 1.0 - dot / (norm_a * norm_b)

        distances = [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        ordered = sorted(distances)
        percentile = max(0.0, min(float(self.breakpoint_threshold), 100.0)) / 100.0
        threshold_idx = min(round(percentile * (len(ordered) - 1)), len(ordered) - 1)
        threshold = ordered[threshold_idx]
        min_piece_len = max(60, self.chunk_size // 4)

        pieces: list[str] = []
        buffer = sentences[0]
        for idx, sentence in enumerate(sentences[1:]):
            boundary = distances[idx] >= threshold and len(buffer) >= min_piece_len
            would_overflow = len(buffer) + len(sentence) > self.chunk_size
            if boundary or would_overflow:
                pieces.extend(self._split_long_paragraph(buffer))
                buffer = sentence
            else:
                buffer += sentence
        if buffer:
            pieces.extend(self._split_long_paragraph(buffer))
        return [piece for piece in pieces if piece.strip()]

    # ── Step 2: 结构分块 (标题 + 段落边界感知) ────────────────

    def _structural_chunk(self, blocks: list[ParseBlock]) -> list[ParseBlock]:
        """按标题/段落结构切分, 长段落内再按字符切"""
        result: list[ParseBlock] = []
        for block in blocks:
            if block.is_table or len(block.content) <= self.chunk_size:
                result.append(block)
                continue

            for piece in self._split_long_paragraph(block.content):
                result.append(ParseBlock(
                    kind=block.kind,
                    content=piece,
                    page=block.page,
                    heading_path=list(block.heading_path),
                    is_table=False,
                    is_image=block.is_image,
                    ocr_text=block.ocr_text,
                ))
        return result

    def _split_long_paragraph(self, text: str) -> list[str]:
        """优先按段落/句子边界切, 最后才硬切字符"""
        pieces: list[str] = []

        # 1. 先按空行切
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        buffer = ""
        for section in sections:
            if len(section) > self.chunk_size:
                # 2. 过长的节再按句子切
                for sent_piece in self._split_by_sentences(section):
                    if buffer:
                        if len(buffer) + len(sent_piece) <= self.chunk_size:
                            buffer += "\n" + sent_piece
                        else:
                            pieces.append(buffer)
                            buffer = sent_piece
                    else:
                        buffer = sent_piece
            else:
                if len(buffer) + len(section) + 1 <= self.chunk_size:
                    buffer = (buffer + "\n" + section).strip()
                else:
                    if buffer:
                        pieces.append(buffer)
                    buffer = section

        if buffer:
            pieces.append(buffer)
        return pieces or [text[: self.chunk_size]]

    def _split_by_sentences(self, text: str) -> list[str]:
        """按中文句号/英文句点等句子边界切分"""
        import re

        sentences = re.split(r"(?<=[。！？!?；;])\s*", text)
        pieces: list[str] = []
        buffer = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(sent) > self.chunk_size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                # 3. 超长单句: 硬切字符 (最后手段)
                for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                    pieces.append(sent[i : i + self.chunk_size])
            elif len(buffer) + len(sent) <= self.chunk_size:
                buffer += sent
            else:
                pieces.append(buffer)
                buffer = sent
        if buffer:
            pieces.append(buffer)
        return pieces

    # ── Step 3: 固定分块 fallback (旧行为) ────────────────────

    def _fixed_chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            start = end - self.chunk_overlap
        return chunks

    # ── 输出转换 ─────────────────────────────────────────────

    @staticmethod
    def _to_document_chunks(
        blocks: list[ParseBlock],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for idx, block in enumerate(blocks):
            content = block.content.strip()
            if not content and not block.is_image:
                continue
            metadata: dict[str, Any] = {
                "source": source,
                "page": block.page,
                "heading_path": block.heading_str,
                "is_table": block.is_table,
                "is_image": block.is_image,
            }
            if block.ocr_text:
                metadata["ocr_text"] = block.ocr_text
            metadata.update(block.metadata)

            chunks.append(DocumentChunk(
                content=content,
                doc_id=doc_id,
                chunk_index=idx,
                doc_type=doc_type,
                metadata=metadata,
            ))
        return chunks
