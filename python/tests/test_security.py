"""安全边界测试：上传、鉴权与只读图查询。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    import api.main as api_mod
    from services.knowledge_graph import KnowledgeGraphService
    from services.vector_store import VectorStoreService
    from tests.test_api import _build_fake_hitl_ingest_graph

    with pytest.MonkeyPatch.context() as mp:
        async def noop_init(self):
            return None

        async def noop_close(self):
            return None

        mp.setattr(VectorStoreService, "init", noop_init)
        mp.setattr(KnowledgeGraphService, "init", noop_init)
        mp.setattr(KnowledgeGraphService, "close", noop_close)

        from fastapi.testclient import TestClient
        with TestClient(api_mod.app) as test_client:
            api_mod.workflows["ingest"] = _build_fake_hitl_ingest_graph()
            yield test_client, api_mod


def test_readonly_cypher_accepts_queries_and_rejects_writes():
    from services.knowledge_graph import KnowledgeGraphService

    assert KnowledgeGraphService.validate_readonly_cypher(
        "MATCH (e:Entity) RETURN e LIMIT 5"
    ).startswith("MATCH")

    for unsafe in (
        "MATCH (e) DETACH DELETE e",
        "CALL db.labels()",
        "LOAD CSV FROM 'https://example.com/a.csv' AS row RETURN row",
        "MATCH (e) RETURN e; MATCH (n) RETURN n",
    ):
        with pytest.raises(ValueError):
            KnowledgeGraphService.validate_readonly_cypher(unsafe)


def test_relation_type_is_allowlisted():
    from services.knowledge_graph import KnowledgeGraphService

    assert KnowledgeGraphService._normalize_relation_type("works_at") == "WORKS_AT"
    assert KnowledgeGraphService._normalize_relation_type("DELETE]-(n)") == "RELATED_TO"
    assert KnowledgeGraphService._normalize_relation_type("负责") == "RELATED_TO"


def test_write_api_key_is_optional_but_enforced_when_configured(client, monkeypatch):
    from config import settings

    c, _ = client
    monkeypatch.setattr(settings, "api_key", "test-secret")

    denied = c.post("/api/ingest/upload", files={"file": ("a.md", b"# A", "text/markdown")})
    assert denied.status_code == 401

    allowed = c.post(
        "/api/ingest/upload",
        files={"file": ("a.md", b"# A", "text/markdown")},
        headers={"X-API-Key": "test-secret"},
    )
    assert allowed.status_code != 401


def test_upload_uses_generated_name_and_stays_in_upload_dir(client, tmp_path, monkeypatch):
    from config import settings

    c, _ = client
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    response = c.post(
        "/api/ingest/upload",
        files={"file": ("../../escape.md", b"# safe", "text/markdown")},
    )
    assert response.status_code == 200
    saved = list(tmp_path.glob("*.md"))
    assert len(saved) == 1
    assert saved[0].parent == tmp_path
    assert saved[0].name != "escape.md"


def test_upload_size_limit(client, tmp_path, monkeypatch):
    from config import settings

    c, _ = client
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)

    response = c.post(
        "/api/ingest/upload",
        files={"file": ("large.md", b"x" * (1024 * 1024 + 1), "text/markdown")},
    )
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_batch_file_count_limit(client, monkeypatch):
    from config import settings

    c, _ = client
    monkeypatch.setattr(settings, "max_batch_files", 1)
    response = c.post(
        "/api/ingest/batch",
        files=[
            ("files", ("a.md", b"# A", "text/markdown")),
            ("files", ("b.md", b"# B", "text/markdown")),
        ],
    )
    assert response.status_code == 413
