"""评测指标与数据集约束测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_retrieval import EvalSample, evaluate, load_dataset
from benchmarks.agentic_ablation import build_source_aliases, evaluate_variants


@pytest.mark.asyncio
async def test_evaluate_requires_matching_source_document():
    async def search(_query: str, _top_k: int):
        return [({"content": "正确片段", "source": "wrong.md"}, 0.99)]

    result = await evaluate(
        search,
        [EvalSample(question="问题", golden_fragment="正确片段", source_doc="right.md")],
        [1],
        warmup_queries=0,
    )
    assert result.mrr == 0
    assert result.per_k[0].recall == 0


@pytest.mark.asyncio
async def test_evaluate_metrics_with_matching_source():
    async def search(_query: str, _top_k: int):
        return [
            ({"content": "无关", "source": "right.md"}, 0.9),
            ({"content": "包含正确片段的正文", "source": "right.md"}, 0.8),
        ]

    result = await evaluate(
        search,
        [EvalSample(question="问题", golden_fragment="正确片段", source_doc="right.md")],
        [1, 2],
        warmup_queries=0,
    )
    assert result.mrr == 0.5
    assert result.per_k[0].recall == 0
    assert result.per_k[1].recall == 1


def test_dataset_loader_rejects_empty_and_duplicate_rows(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {"question": "Q1", "golden_fragment": "A1", "source_doc": "a.md"},
        {"question": "Q1", "golden_fragment": "A2", "source_doc": "b.md"},
        {"question": "", "golden_fragment": "A3", "source_doc": "c.md"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    assert load_dataset(path) == [EvalSample(question="Q1", golden_fragment="A1", source_doc="a.md")]


@pytest.mark.asyncio
async def test_ablation_uses_same_dataset_and_excludes_warmup_rounds():
    from agents.qa_agent import RetrievedContext

    calls = 0

    async def variant(_question: str, _top_k: int):
        nonlocal calls
        calls += 1
        return [RetrievedContext("正确片段", "right.md", 0.8, "vector")], 2

    dataset = [EvalSample("问题", "正确片段", "right.md")]
    results = await evaluate_variants({"self_rag": variant}, dataset, [1], warmup_queries=1)

    assert calls == 2  # 一次 warm-up + 一次计入指标的查询
    assert results["self_rag"].retrieval.mrr == 1
    assert results["self_rag"].average_retrieval_rounds == 2


@pytest.mark.asyncio
async def test_ablation_maps_managed_upload_id_to_golden_source():
    from agents.qa_agent import RetrievedContext

    async def variant(_question: str, _top_k: int):
        return [RetrievedContext("正确片段", "/app/uploads/abc123.md", 0.8, "vector")], 1

    dataset = [EvalSample("问题", "正确片段", "right.md")]
    aliases = build_source_aliases(
        [{"file_id": "abc123.md", "original_name": "right.md", "status": "success"}],
        dataset,
    )
    result = await evaluate_variants(
        {"vector": variant}, dataset, [1], warmup_queries=0, source_aliases=aliases,
    )
    assert result["vector"].retrieval.per_k[0].recall == 1


def test_ablation_rejects_missing_or_ambiguous_uploaded_sources():
    dataset = [EvalSample("问题", "正确片段", "right.md")]
    with pytest.raises(ValueError, match="缺失"):
        build_source_aliases([], dataset)
    with pytest.raises(ValueError, match="同名多份"):
        build_source_aliases(
            [
                {"file_id": "a.md", "original_name": "right.md", "status": "success"},
                {"file_id": "b.md", "original_name": "right.md", "status": "success"},
            ],
            dataset,
        )
