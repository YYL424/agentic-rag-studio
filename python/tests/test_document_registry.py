"""Persistent document catalog tests."""

from services.document_registry import DocumentRegistry


def test_registry_persists_and_deduplicates_by_content_hash(tmp_path):
    path = tmp_path / "documents.sqlite3"
    registry = DocumentRegistry(path)
    registry.init()

    first, created = registry.register(
        file_id="a.md",
        original_name="original.md",
        content_sha256="abc",
        file_size=10,
        thread_id="ingest-a",
    )
    duplicate, duplicate_created = registry.register(
        file_id="b.md",
        original_name="copy.md",
        content_sha256="abc",
        file_size=10,
        thread_id="ingest-b",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate["file_id"] == first["file_id"] == "a.md"

    registry.update("a.md", status="success", chunks_count=2)
    reopened = DocumentRegistry(path)
    reopened.init()
    assert reopened.get("a.md")["status"] == "success"
    assert reopened.list_documents()[0]["chunks_count"] == 2

    reopened.delete("a.md")
    assert reopened.list_documents() == []
