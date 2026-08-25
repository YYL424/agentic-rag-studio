"""
知识更新 Agent — 监听文档变更，增量更新向量库和知识图谱

核心能力:
  1. 文件系统监听 (Watchdog) / Kafka CDC 消费
  2. 真增量 Diff (difflib): 文档局部修改只重新处理变化 chunk
     - 旧实现: 删全文档向量再重建 (假增量, 分钟级)
     - 新实现: 内容级 diff, 只处理变化的 chunk
  3. 快照管理: 持久化上次入库的 chunk 内容, 用于 diff 对比
  4. 版本管理：知识节点带时间戳和版本号
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    chunks_reprocessed: int = 0    # 真增量: 实际重新处理的 chunk 数
    chunks_unchanged: int = 0      # 未变化直接复用的 chunk 数
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """
    知识更新 Agent

    支持两种模式:
      1. 文件监听模式 (Watchdog): 监听本地文件系统变更
      2. CDC 模式 (Kafka): 消费来自消息队列的变更事件

    工作流:
      detect_change → diff_analysis (difflib) → incremental_parse → update_vector_store → update_knowledge_graph → log
    """

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._file_hashes: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}
        # 快照: doc_id → {"chunk_ids": [...], "contents": [...]}
        self._snapshots: dict[str, dict] = {}
        self._snapshot_dir = os.path.join(settings.upload_dir, ".snapshots")

    # ── public API ───────────────────────────────────────────

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        """处理单个文档变更"""
        start = time.time()
        result = UpdateResult(change=change)

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        """批量处理文档变更"""
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        """扫描文件列表，检测变更"""
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            self._file_hashes[fp] = new_hash

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
            ))
            del self._file_hashes[fp]

        return changes

    # ── watchdog mode ────────────────────────────────────────

    def start_watching(self, directory: str) -> None:
        """启动文件系统监听（非阻塞，在独立线程运行）"""
        import threading
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                    asyncio.run(agent.process_change(change))

            def on_modified(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                    asyncio.run(agent.process_change(change))

            def on_deleted(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.DELETED)
                    asyncio.run(agent.process_change(change))

        observer = Observer()
        observer.schedule(_Handler(), directory, recursive=True)

        def _run():
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── kafka CDC mode ───────────────────────────────────────

    async def start_kafka_consumer(self) -> None:
        """启动 Kafka CDC 消费者"""
        import json
        from confluent_kafka import Consumer

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "knowledge-update-agent",
            "auto.offset.reset": "latest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([settings.kafka_topic_doc_changes])

        try:
            while True:
                msg = await asyncio.to_thread(consumer.poll, 1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                payload = json.loads(msg.value().decode("utf-8"))
                change = DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType(payload["change_type"]),
                    old_hash=payload.get("old_hash", ""),
                    new_hash=payload.get("new_hash", ""),
                )
                await self.process_change(change)
        finally:
            consumer.close()

    # ── internal handlers ────────────────────────────────────

    async def _handle_create(
        self,
        change: DocumentChange,
        result: UpdateResult,
        chunks: list | None = None,
    ) -> None:
        if not self.doc_parser:
            return
        if chunks is None:
            chunks = await self.doc_parser.parse(change.file_path)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)

        if self.knowledge_extractor and self.knowledge_graph:
            extractions = await self.knowledge_extractor.extract(chunks)
            chunk_sources = {c.chunk_id: c.metadata.get("source", change.file_path) for c in chunks}
            for ext in extractions:
                source = chunk_sources.get(ext.source_chunk_id, change.file_path)
                for ent in ext.entities:
                    version = self._bump_version(ent.name)
                    await self.knowledge_graph.upsert_entity(
                        ent,
                        version=version,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
                    result.entities_added += 1
                for rel in ext.relations:
                    await self.knowledge_graph.add_relation(
                        rel,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
                    result.relations_added += 1

        result.chunks_reprocessed = len(chunks)
        self._save_snapshot(change.file_path, chunks)

    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        """
        真增量更新 (文本级 diff):
          1. 解析新版本 → 新 chunk 列表
          2. 与快照对比 (difflib 内容匹配):
             - 未变化的 chunk → 直接复用旧向量和图谱节点 (0 开销)
             - 新增/修改的 chunk → 删除旧向量, 重新向量化 + 重新抽取
             - 消失的 chunk → 删除旧向量
          3. 更新快照
        """
        if not self.doc_parser:
            return

        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]
        old_snapshot = self._load_snapshot(doc_id)
        new_chunks = await self.doc_parser.parse(change.file_path)

        # 快照不存在 → 退化为全量重建 (与 create 相同)
        if not old_snapshot:
            await self._handle_create(change, result, chunks=new_chunks)
            return

        old_by_id = dict(zip(old_snapshot["chunk_ids"], old_snapshot["contents"]))
        new_contents = [c.content for c in new_chunks]

        # 以 chunk_id + content 双重匹配。不能只用 set(content)：重复段落会被
        # 合并，而且前文插入导致索引变化后，快照 ID 会与实际向量 ID 脱节。
        new_by_id = {c.chunk_id: c.content for c in new_chunks}
        stale_ids = [cid for cid, content in old_by_id.items() if new_by_id.get(cid) != content]

        # ── 2. 新增/修改的 chunk: 只处理这些 ──────────────────
        changed_chunks = [c for c in new_chunks if old_by_id.get(c.chunk_id) != c.content]
        unchanged_chunks = [c for c in new_chunks if old_by_id.get(c.chunk_id) == c.content]

        # ── 3. difflib 行级 diff (用于结果报告 + change.diff_chunks) ──
        diff_report = self._compute_diff(old_snapshot["contents"], new_contents)
        change.diff_chunks = diff_report["changed_lines_summary"]

        # ── 4. 向量库: 只动变化的 ─────────────────────────────
        if self.vector_store:
            if stale_ids:
                await self.vector_store.delete_by_chunk_ids(stale_ids)
                result.vectors_deleted = len(stale_ids)
            if changed_chunks:
                await self.vector_store.add_chunks(changed_chunks)
                result.vectors_added = len(changed_chunks)

        # 图谱 provenance 与向量使用同一批 stale chunk，先移除旧事实，
        # 再写入变化 chunk，避免文档修改后仍检索到过期实体/关系。
        if self.knowledge_graph and stale_ids:
            await self.knowledge_graph.delete_by_chunk_ids(stale_ids)

        # ── 5. 图谱: 只对变化的 chunk 重新抽取 ────────────────
        if self.knowledge_extractor and self.knowledge_graph and changed_chunks:
            extractions = await self.knowledge_extractor.extract(changed_chunks)
            chunk_sources = {c.chunk_id: c.metadata.get("source", change.file_path) for c in changed_chunks}
            for ext in extractions:
                source = chunk_sources.get(ext.source_chunk_id, change.file_path)
                for ent in ext.entities:
                    version = self._bump_version(ent.name)
                    await self.knowledge_graph.upsert_entity(
                        ent,
                        version=version,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
                    result.entities_updated += 1
                for rel in ext.relations:
                    await self.knowledge_graph.add_relation(
                        rel,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                    )
                    result.relations_added += 1

        result.chunks_reprocessed = len(changed_chunks)
        result.chunks_unchanged = len(unchanged_chunks)

        # ── 6. 更新快照 ───────────────────────────────────────
        self._save_snapshot(change.file_path, new_chunks)

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)

        self._remove_snapshot(doc_id)

    # ── diff computation (difflib) ───────────────────────────

    @staticmethod
    def _compute_diff(old_contents: list[str], new_contents: list[str]) -> dict:
        """
        行级 diff (difflib.SequenceMatcher):
        返回变化行统计 + 变更摘要 (精确的行级增删, 非集合对比)
        """
        old_text = "\n".join(old_contents).splitlines()
        new_text = "\n".join(new_contents).splitlines()

        matcher = difflib.SequenceMatcher(a=old_text, b=new_text, autojunk=False)
        added_lines: list[str] = []
        removed_lines: list[str] = []
        changed_blocks = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed_blocks += 1
            if tag in ("replace", "delete"):
                removed_lines.extend(old_text[i1:i2])
            if tag in ("replace", "insert"):
                added_lines.extend(new_text[j1:j2])

        total = max(len(old_text) + len(new_text), 1)
        change_ratio = (len(added_lines) + len(removed_lines)) / total

        # 变更摘要: 前 5 行变化
        summary = [f"+ {l}" for l in added_lines[:5]] + [f"- {l}" for l in removed_lines[:5]]

        return {
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "added_count": len(added_lines),
            "removed_count": len(removed_lines),
            "changed_blocks": changed_blocks,
            "change_ratio": round(change_ratio, 4),
            "is_major_change": change_ratio > 0.3,
            "changed_lines_summary": summary,
        }

    # ── snapshot management (快照持久化) ─────────────────────

    def _snapshot_path(self, doc_id: str) -> str:
        return os.path.join(self._snapshot_dir, f"{doc_id}.json")

    def _save_snapshot(self, file_path: str, chunks: list) -> None:
        """保存快照: 下次 diff 的对比基准"""
        doc_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        snapshot = {
            "file_path": file_path,
            "chunk_ids": [c.chunk_id for c in chunks],
            "contents": [c.content for c in chunks],
            "saved_at": time.time(),
        }
        self._snapshots[doc_id] = snapshot

        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            with open(self._snapshot_path(doc_id), "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)
        except OSError as e:
            logger.warning("snapshot persist failed: %s", e)

    def _load_snapshot(self, doc_id: str) -> dict | None:
        if doc_id in self._snapshots:
            return self._snapshots[doc_id]
        try:
            with open(self._snapshot_path(doc_id), encoding="utf-8") as f:
                snapshot = json.load(f)
            self._snapshots[doc_id] = snapshot
            return snapshot
        except (OSError, json.JSONDecodeError):
            return None

    def _remove_snapshot(self, doc_id: str) -> None:
        self._snapshots.pop(doc_id, None)
        try:
            os.remove(self._snapshot_path(doc_id))
        except OSError:
            pass

    # ── utilities ────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    def _bump_version(self, entity_name: str) -> int:
        ver = self._version_counter.get(entity_name, 0) + 1
        self._version_counter[entity_name] = ver
        return ver
