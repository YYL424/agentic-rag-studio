"""Agentic retrieval ablation benchmark on an already populated knowledge base.

The variants isolate the contribution of query rewriting, GraphRAG/RRF and
Self-RAG while reusing the same dataset, stores and retrieval metrics.

Example (run from ``python/`` after indexing the evaluation documents):

    python benchmarks/agentic_ablation.py \
      --dataset benchmarks/golden_set.jsonl \
      --top-k 1,3,5 \
      --out benchmarks/results/agentic_ablation.json

This benchmark evaluates retrieved evidence, not generated-answer correctness.
Query rewriting and Self-RAG use the configured LLM and may incur API cost.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.qa_agent import QAAgent, RetrievedContext  # noqa: E402
from benchmarks.eval_retrieval import EvalResult, EvalSample, evaluate, load_dataset  # noqa: E402
from config import settings  # noqa: E402


VariantSearch = Callable[[str, int], Awaitable[tuple[list[RetrievedContext], int]]]


@dataclass
class VariantResult:
    retrieval: EvalResult
    average_retrieval_rounds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.retrieval.as_dict(),
            "average_retrieval_rounds": round(self.average_retrieval_rounds, 3),
        }


async def evaluate_variants(
    variants: dict[str, VariantSearch],
    dataset: list[EvalSample],
    top_k_list: list[int] | tuple[int, ...],
    *,
    warmup_queries: int = 2,
) -> dict[str, VariantResult]:
    """Evaluate all variants under the same metric and warm-up protocol."""
    results: dict[str, VariantResult] = {}
    warmup_count = min(max(0, warmup_queries), len(dataset))

    for name, variant in variants.items():
        rounds_observed: list[int] = []

        async def search(question: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
            contexts, rounds = await variant(question, top_k)
            rounds_observed.append(rounds)
            return [
                (
                    {
                        "content": context.content,
                        "source": context.source,
                        "metadata": context.metadata,
                    },
                    context.score,
                )
                for context in contexts[:top_k]
            ]

        retrieval = await evaluate(
            search,
            dataset,
            top_k_list,
            warmup_queries=warmup_queries,
        )
        measured_rounds = rounds_observed[warmup_count:]
        average_rounds = sum(measured_rounds) / len(measured_rounds) if measured_rounds else 0.0
        results[name] = VariantResult(retrieval, average_rounds)

    return results


def build_variants(
    agent: QAAgent,
    *,
    reranker: Any = None,
) -> dict[str, VariantSearch]:
    """Build comparable retrieval variants around one configured QA agent."""

    async def vector_raw(question: str, top_k: int) -> tuple[list[RetrievedContext], int]:
        rows = await agent.vector_store.search(question, top_k=top_k)
        contexts = [
            RetrievedContext(
                content=doc.get("content", ""),
                source=doc.get("source", "vector_store"),
                score=score,
                retrieval_type="vector",
                metadata=dict(doc.get("metadata", {})),
            )
            for doc, score in rows
        ]
        return contexts, 1

    async def rewritten_vector(question: str, top_k: int) -> tuple[list[RetrievedContext], int]:
        rewritten = await agent._rewrite_query(question)
        queries = rewritten.get("queries") or [question]
        ranked_lists = await asyncio.gather(*(
            agent._vector_retrieve({"queries": [query]}) for query in queries[:3]
        ))
        return agent._reciprocal_rank_fusion(list(ranked_lists))[:top_k], 1

    async def hybrid_rrf(question: str, top_k: int) -> tuple[list[RetrievedContext], int]:
        rewritten = await agent._rewrite_query(question)
        previous = agent.reranker
        agent.reranker = None
        try:
            return await agent.retrieve(question, rewritten, top_k=top_k), 1
        finally:
            agent.reranker = previous

    async def hybrid_rrf_self_rag(question: str, top_k: int) -> tuple[list[RetrievedContext], int]:
        rewritten = await agent._rewrite_query(question)
        previous_reranker = agent.reranker
        previous_self_rag = settings.enable_self_rag
        agent.reranker = None
        settings.enable_self_rag = True
        try:
            initial = await agent.retrieve(question, rewritten, top_k=max(top_k, 8))
            contexts, rounds, _steps = await agent.refine_retrieval(question, initial, rounds=1)
            return contexts[:top_k], rounds
        finally:
            agent.reranker = previous_reranker
            settings.enable_self_rag = previous_self_rag

    variants: dict[str, VariantSearch] = {
        "vector_raw": vector_raw,
        "vector_rewritten": rewritten_vector,
        "hybrid_rrf": hybrid_rrf,
        "hybrid_rrf_self_rag": hybrid_rrf_self_rag,
    }

    if reranker is not None:
        async def hybrid_rrf_reranker(question: str, top_k: int) -> tuple[list[RetrievedContext], int]:
            rewritten = await agent._rewrite_query(question)
            previous = agent.reranker
            agent.reranker = reranker
            try:
                return await agent.retrieve(question, rewritten, top_k=top_k), 1
            finally:
                agent.reranker = previous

        variants["hybrid_rrf_reranker"] = hybrid_rrf_reranker

    return variants


def _dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _print_results(results: dict[str, VariantResult], top_k: list[int]) -> None:
    columns = ["variant", "MRR", *(f"R@{k}" for k in top_k), "P95(ms)", "avg rounds"]
    print("\n" + " | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for name, result in results.items():
        recalls = {metric.k: metric.recall for metric in result.retrieval.per_k}
        values = [
            name,
            f"{result.retrieval.mrr:.4f}",
            *(f"{recalls[k]:.4f}" for k in top_k),
            f"{result.retrieval.latency_p95_ms:.1f}",
            f"{result.average_retrieval_rounds:.2f}",
        ]
        print(" | ".join(values))


async def _run(args: argparse.Namespace) -> None:
    from services.knowledge_graph import KnowledgeGraphService
    from services.vector_store import VectorStoreService

    dataset = load_dataset(args.dataset)
    if not dataset:
        raise SystemExit("评测数据集为空")

    vector_store = VectorStoreService()
    needs_graph = any(name.startswith("hybrid_") for name in args.variants)
    knowledge_graph = KnowledgeGraphService() if needs_graph else None
    await vector_store.init()
    try:
        if knowledge_graph is not None:
            await knowledge_graph.init()
        agent = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph, reranker=False)
        agent.reranker = None

        reranker = None
        if "hybrid_rrf_reranker" in args.variants:
            from services.reranker import RerankerService
            candidate = RerankerService()
            if not candidate.available:
                raise SystemExit("请求了 reranker 变量，但本地 reranker 模型不可用")
            reranker = candidate

        available = build_variants(agent, reranker=reranker)
        unknown = sorted(set(args.variants) - set(available))
        if unknown:
            raise SystemExit(f"未知或不可用的评测变量: {', '.join(unknown)}")
        selected = {name: available[name] for name in args.variants}
        results = await evaluate_variants(
            selected,
            dataset,
            args.top_k,
            warmup_queries=args.warmup_queries,
        )
    finally:
        await vector_store.close()
        if knowledge_graph is not None:
            await knowledge_graph.close()

    _print_results(results, args.top_k)
    payload = {
        "config": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "dataset_sha256": _dataset_sha256(args.dataset),
            "n_samples": len(dataset),
            "top_k": args.top_k,
            "variants": args.variants,
            "vector_store": settings.vector_store_type,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": (
                settings.local_embedding_model
                if settings.embedding_backend == "local"
                else settings.embedding_model
            ),
            "llm_model": settings.openai_model,
            "self_rag_threshold": settings.self_rag_threshold,
            "self_rag_max_rounds": settings.self_rag_max_rounds,
            "python": platform.python_version(),
            "scope": "retrieval_evidence_only",
        },
        "results": {name: result.as_dict() for name, result in results.items()},
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG / Self-RAG 消融评测")
    parser.add_argument("--dataset", required=True, help="人工审核 golden set JSONL")
    parser.add_argument("--top-k", default="1,3,5", help="逗号分隔的 K 值")
    parser.add_argument(
        "--variants",
        default="vector_raw,vector_rewritten,hybrid_rrf,hybrid_rrf_self_rag",
        help="逗号分隔变量；可额外加入 hybrid_rrf_reranker",
    )
    parser.add_argument("--warmup-queries", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/agentic_ablation.json")
    args = parser.parse_args()
    args.top_k = [int(value) for value in args.top_k.split(",") if value.strip()]
    args.variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
