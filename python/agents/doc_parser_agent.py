"""
文档解析 Agent — 多模态文档解析，支持 PDF / Word / PPT / 图片 / 表格 / 纯文本

核心能力:
  1. PDF 解析（marker-pdf 结构化 → pymupdf 表格/排版 → pypdf 降级）
  2. Office 文档解析（docling → python-docx / python-pptx）
  3. 图片 OCR（marker surya → pytesseract）+ LLM 视觉理解
  4. 语义边界分块（相邻句向量距离 → 结构分块 → 固定分块）
  5. 结构化元数据标注（heading_path / is_table / is_image / page）
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from schema import DocType, DocumentChunk  # noqa: F401  (re-export 保持旧导入兼容)
from services.document_parser import UnifiedDocumentParser
from utils.chunker import DocumentChunker


class DocParserAgent:
    """
    文档解析 Agent

    工作流:
      classify → unified_parse (marker/docling/pymupdf) → semantic_chunk → enrich_metadata → output
    """

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".docx": DocType.WORD,
        ".doc": DocType.WORD,
        ".pptx": DocType.PPT,
        ".ppt": DocType.PPT,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".bmp": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
    }

    def __init__(self, embeddings: Any = None) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.parser = UnifiedDocumentParser()
        self.chunker = DocumentChunker(
            embeddings=embeddings,
            semantic_enabled=settings.semantic_chunk_enabled,
            breakpoint_threshold=settings.chunk_breakpoint_threshold,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── public API ───────────────────────────────────────────

    async def parse(self, file_path: str) -> list[DocumentChunk]:
        """解析单个文件，返回文档块列表"""
        doc_type = self._classify(file_path)
        doc_id = self._make_doc_id(file_path)

        # 统一解析引擎: 可选重型解析器 + 轻量解析器降级链
        blocks = await self._run_parse(file_path, doc_type)

        # 图片无 OCR 文本时, LLM 视觉兜底
        blocks = await self._vision_fallback(blocks, file_path, doc_type)

        # 语义分块可能执行 embedding/CPU 计算，放入 worker 避免阻塞 API 事件循环。
        import asyncio
        return await asyncio.to_thread(
            self.chunker.chunk,
            blocks,
            doc_id,
            doc_type,
            file_path,
        )

    async def parse_batch(self, file_paths: list[str]) -> list[DocumentChunk]:
        """批量解析多个文件"""
        import asyncio

        semaphore = asyncio.Semaphore(settings.max_batch_files)

        async def parse_one(file_path: str) -> list[DocumentChunk]:
            async with semaphore:
                return await self.parse(file_path)

        batches = await asyncio.gather(*(parse_one(fp) for fp in file_paths))
        return [chunk for batch in batches for chunk in batch]

    # ── classification ───────────────────────────────────────

    def _classify(self, file_path: str) -> DocType:
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    # ── unified parsing ──────────────────────────────────────

    @staticmethod
    async def _run_parse(file_path: str, doc_type: DocType):
        """在 worker 线程中执行解析 (PDF 解析是 CPU 密集操作)"""
        import asyncio
        return await asyncio.to_thread(DocParserAgent._parse_sync, file_path, doc_type)

    @staticmethod
    def _parse_sync(file_path: str, doc_type: DocType):
        parser = UnifiedDocumentParser()
        try:
            return parser.parse(file_path, doc_type)
        except Exception:
            return []

    async def _vision_fallback(self, blocks, file_path: str, doc_type: DocType):
        """当图片无法 OCR 时, 调用 LLM 多模态能力描述图片"""
        if doc_type != DocType.IMAGE:
            return blocks

        has_text = any(b.content.strip() for b in blocks if not b.is_image)
        if has_text:
            return blocks

        try:
            from PIL import Image
            img = Image.open(file_path)
            description = await self._describe_image_with_llm(img)
            if description:
                from services.document_parser import ParseBlock
                blocks.append(ParseBlock(
                    kind="paragraph",
                    content=description,
                    is_image=True,
                    metadata={"vision_understood": True},
                ))
        except Exception:
            pass
        return blocks

    async def _describe_image_with_llm(self, image: Any) -> str:
        """调用 LLM 多模态能力描述图片内容"""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            SystemMessage(content="你是一个专业的文档分析助手，请详细描述图片中的内容，包括文字、表格、图表信息。"),
            HumanMessage(content=[
                {"type": "text", "text": "请描述这张图片的所有内容："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            return resp.content or ""
        except Exception:
            return ""
