"""SQLite-backed document catalog and upload idempotency registry."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class DocumentRegistry:
    """Persist document metadata beside the managed upload volume."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()

    def init(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    file_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunks_count INTEGER NOT NULL DEFAULT 0,
                    entities_count INTEGER NOT NULL DEFAULT 0,
                    relations_count INTEGER NOT NULL DEFAULT 0,
                    thread_id TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_thread ON documents(thread_id)")

    def register(
        self,
        *,
        file_id: str,
        original_name: str,
        content_sha256: str,
        file_size: int,
        thread_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Create a processing record; return an existing record on duplicate content."""
        now = self._now()
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        file_id, original_name, content_sha256, file_size, status,
                        thread_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'processing', ?, ?, ?)
                    """,
                    (file_id, original_name, content_sha256, file_size, thread_id, now, now),
                )
        except sqlite3.IntegrityError:
            existing = self.get_by_hash(content_sha256)
            if existing is None:
                raise
            return existing, False
        created = self.get(file_id)
        if created is None:
            raise RuntimeError("document registry insert was not persisted")
        return created, True

    def get(self, file_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM documents WHERE file_id = ?", (file_id,))

    def get_by_hash(self, content_sha256: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM documents WHERE content_sha256 = ?",
            (content_sha256,),
        )

    def get_by_thread_id(self, thread_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM documents WHERE thread_id = ?", (thread_id,))

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update(self, file_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "chunks_count",
            "entities_count",
            "relations_count",
            "thread_id",
            "error_message",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = self._now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [*updates.values(), file_id]
        with self._connection() as conn:
            conn.execute(f"UPDATE documents SET {assignments} WHERE file_id = ?", values)

    def delete(self, file_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM documents WHERE file_id = ?", (file_id,))

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
