"""Embedding 工厂的轻量回退测试；不下载真实模型。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.embeddings import LocalEmbeddings, _download_model_file  # noqa: E402


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


def test_local_embeddings_prefers_sentence_transformers(monkeypatch):
    module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        model_name = ""

        def __init__(self, model_name: str) -> None:
            FakeSentenceTransformer.model_name = model_name

        def encode(self, value, normalize_embeddings: bool):
            assert normalize_embeddings is True
            if isinstance(value, list):
                return _FakeVector([[1.0, 0.0] for _ in value])
            return _FakeVector([0.0, 1.0])

    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embeddings = LocalEmbeddings("test/bge")

    assert embeddings._backend == "sentence-transformers/test/bge"
    assert FakeSentenceTransformer.model_name == "test/bge"
    assert embeddings.embed_documents(["甲", "乙"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embeddings.embed_query("问题") == [0.0, 1.0]


def test_local_embeddings_falls_back_to_chroma_onnx(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    for package_name in ("chromadb", "chromadb.utils", "chromadb.utils.embedding_functions"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    module_name = "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"
    onnx_module = ModuleType(module_name)

    class FakeOnnxModel:
        _MODEL_SHA256 = "unused"

        def __call__(self, texts: list[str]):
            return [_FakeVector([float(len(text)), 0.5]) for text in texts]

    onnx_module.ONNXMiniLM_L6_V2 = FakeOnnxModel
    monkeypatch.setitem(sys.modules, module_name, onnx_module)

    embeddings = LocalEmbeddings("missing/bge")

    assert embeddings._backend == "chroma-onnx/all-MiniLM-L6-v2"
    assert embeddings.embed_documents([]) == []
    assert embeddings.embed_documents(["知识", "RAG"]) == [[2.0, 0.5], [3.0, 0.5]]
    assert embeddings.embed_query("Agent") == [5.0, 0.5]


def test_model_download_retries_and_verifies_hash(tmp_path, monkeypatch):
    content = b"verified model bytes"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    calls = 0

    class FakeResponse:
        def __enter__(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("simulated TLS timeout")
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size: int):
            assert chunk_size == 64 * 1024
            yield content

    httpx_module = ModuleType("httpx")
    httpx_module.Timeout = lambda **kwargs: kwargs
    httpx_module.stream = lambda *_args, **_kwargs: FakeResponse()
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)

    destination = tmp_path / "onnx.tar.gz"
    _download_model_file("https://example.invalid/model", str(destination), expected_sha256)

    assert calls == 2
    assert destination.read_bytes() == content
    assert not (tmp_path / "onnx.tar.gz.part").exists()
