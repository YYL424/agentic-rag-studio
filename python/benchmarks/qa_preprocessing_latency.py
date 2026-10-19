"""Repeatable offline comparison of sequential and parallel QA graph stages.

The fake agent sleeps to emulate model and retrieval I/O. Its measurements are
for this controlled workflow only; they are not production API latency or
answer-quality measurements. Both variants run through the real LangGraph.

Run from python/:
    python benchmarks/qa_preprocessing_latency.py \
      --dataset benchmarks/golden_set.jsonl \
      --out ../docs/evidence/qa_preprocessing_latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.qa_agent import QAResult, QueryIntent, RetrievedContext  # noqa: E402
from benchmarks.eval_retrieval import _percentile, load_dataset  # noqa: E402
from orchestrator.graph import _build_qa_graph  # noqa: E402


@dataclass(frozen=True)
class StageDelays:
    classify_ms: float = 40.0
    rewrite_ms: float = 60.0
    retrieve_ms: float = 15.0
    self_rag_ms: float = 8.0
    generate_ms: float = 25.0


class SimulatedQAAgent:
    def __init__(self, answers: dict[str, tuple[str, str]], delays: StageDelays) -> None:
        self.answers = answers
        self.delays = delays
        self.calls = {name: 0 for name in ("classify", "rewrite", "retrieve", "self_rag", "generate")}

    async def _classify_intent(self, _question: str) -> QueryIntent:
        self.calls["classify"] += 1
        await asyncio.sleep(self.delays.classify_ms / 1000)
        return QueryIntent.FACTOID

    async def _rewrite_query(self, question: str) -> dict:
        self.calls["rewrite"] += 1
        await asyncio.sleep(self.delays.rewrite_ms / 1000)
        return {"queries": [question], "entities": [], "keywords": []}

    async def retrieve(self, question: str, rewritten: dict, top_k: int) -> list[RetrievedContext]:
        self.calls["retrieve"] += 1
        assert rewritten["queries"] == [question] and top_k == 8
        await asyncio.sleep(self.delays.retrieve_ms / 1000)
        fragment, source = self.answers[question]
        return [RetrievedContext(fragment, source, 0.8, "vector")]

    async def refine_retrieval(
        self, _question: str, contexts: list[RetrievedContext], *, rounds: int,
    ) -> tuple[list[RetrievedContext], int, list[str]]:
        self.calls["self_rag"] += 1
        await asyncio.sleep(self.delays.self_rag_ms / 1000)
        return contexts, rounds, ["simulated"]

    async def build_result(
        self, question: str, intent: QueryIntent, contexts: list[RetrievedContext],
        rounds: int, reasoning: list[str],
    ) -> QAResult:
        self.calls["generate"] += 1
        await asyncio.sleep(self.delays.generate_ms / 1000)
        return QAResult(question, contexts[0].content, contexts, intent, 0.8, reasoning, rounds)


def _summary(latencies: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(_percentile(latencies, 95), 3),
    }


async def compare_workflows(
    questions: list[str],
    answers: dict[str, tuple[str, str]],
    delays: StageDelays,
    *,
    runs: int = 3,
    warmup: int = 2,
) -> dict:
    if not questions or runs < 1 or warmup < 0:
        raise ValueError("questions must be nonempty, runs >= 1, warmup >= 0")

    agents = {
        name: SimulatedQAAgent(answers, delays)
        for name in ("sequential", "parallel")
    }
    graphs = {
        name: _build_qa_graph(
            agents[name], checkpointer=None, parallel_preprocessing=(name == "parallel"),
        )
        for name in agents
    }
    latencies = {name: [] for name in graphs}
    expected_results: dict[str, tuple[str, str, str]] = {}
    parity_checked = 0

    for name, graph in graphs.items():
        for question in questions[:warmup]:
            await graph.ainvoke({"question": question})

    for run_index in range(runs):
        order = ("sequential", "parallel") if run_index % 2 == 0 else ("parallel", "sequential")
        for question in questions:
            for name in order:
                start = time.perf_counter()
                state = await graphs[name].ainvoke({"question": question})
                latencies[name].append((time.perf_counter() - start) * 1000)
                result: QAResult = state["result"]
                observed = (result.answer, result.contexts[0].source, result.intent.value)
                if name == "sequential":
                    expected_results[question] = observed
                elif observed != expected_results[question]:
                    raise AssertionError(f"workflow outputs differ for {question!r}")
                if name == "parallel":
                    parity_checked += 1

    sequential = _summary(latencies["sequential"])
    parallel = _summary(latencies["parallel"])
    return {
        "sequential": sequential,
        "parallel": parallel,
        "p50_saved_ms": round(sequential["p50_ms"] - parallel["p50_ms"], 3),
        "p50_reduction_pct": round(
            100 * (1 - parallel["p50_ms"] / sequential["p50_ms"]), 2,
        ),
        "p95_saved_ms": round(sequential["p95_ms"] - parallel["p95_ms"], 3),
        "output_parity_checks": parity_checked,
        "calls_per_workflow": {name: agent.calls for name, agent in agents.items()},
    }


async def _run(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset)
    samples = load_dataset(dataset_path)[:args.samples]
    if not samples:
        raise SystemExit("No valid evaluation questions found")
    questions = [sample.question for sample in samples]
    answers = {sample.question: (sample.golden_fragment, sample.source_doc) for sample in samples}
    delays = StageDelays(
        classify_ms=args.classify_ms,
        rewrite_ms=args.rewrite_ms,
        retrieve_ms=args.retrieve_ms,
        self_rag_ms=args.self_rag_ms,
        generate_ms=args.generate_ms,
    )
    if any(value < 0 for value in asdict(delays).values()):
        raise SystemExit("Simulated stage delays must be nonnegative")

    result = await compare_workflows(questions, answers, delays, runs=args.runs, warmup=args.warmup)
    payload = {
        "scope": "offline_simulated_io_workflow_latency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "questions": len(questions),
        "runs": args.runs,
        "warmup_queries_per_variant": min(args.warmup, len(questions)),
        "simulated_stage_delays_ms": asdict(delays),
        "python": platform.python_version(),
        "os": platform.system(),
        "measurements": result,
        "limitations": (
            "Uses asyncio.sleep and a deterministic fake agent. No paid LLM, real database, "
            "answer accuracy, token savings, or production latency is measured."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Sequential P50 {result['sequential']['p50_ms']:.1f} ms; "
        f"parallel P50 {result['parallel']['p50_ms']:.1f} ms; "
        f"saved {result['p50_saved_ms']:.1f} ms ({result['p50_reduction_pct']:.1f}%)."
    )
    print(f"Output parity: {result['output_parity_checks']} comparisons. Report: {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline simulated I/O latency of the QA graph")
    parser.add_argument("--dataset", default="benchmarks/golden_set.jsonl")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--classify-ms", type=float, default=40)
    parser.add_argument("--rewrite-ms", type=float, default=60)
    parser.add_argument("--retrieve-ms", type=float, default=15)
    parser.add_argument("--self-rag-ms", type=float, default=8)
    parser.add_argument("--generate-ms", type=float, default=25)
    parser.add_argument("--out", default="../docs/evidence/qa_preprocessing_latency.json")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
