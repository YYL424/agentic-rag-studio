"""
RAG 检索质量评测

评测指标:
  1. Recall@K: 正确答案 chunk 是否出现在 Top-K 检索结果中
  2. MRR (Mean Reciprocal Rank): 正确答案的平均倒数排名
  3. NDCG@K: 考虑排名的相关性折扣
  4. 检索延迟 P50/P95

数据集格式 (JSONL), 每行:
  {"question": "...", "golden_fragment": "...", "source_doc": "..."}

判定命中: 检索到的 chunk 内容包含 golden_fragment (归一化去空白后子串匹配,
          或最长公共子串占比 ≥ 60%, 容忍分块边界差异)

使用方法:
  1. 准备评测数据集 (见 benchmarks/generate_qa_pairs.py)
  2. 先入库评测文档
  3. 运行: python benchmarks/eval_retrieval.py --dataset qa_pairs.jsonl --top-k 5,10

编程接口:
  await evaluate(search_fn, dataset, top_k_list) -> EvalResult
  search_fn: async (query, top_k) -> list[tuple[dict, float]]
             dict 需含 "content" 字段
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 数据结构 ───────────────────────────────────────────────

@dataclass
class EvalSample:
    question: str
    golden_fragment: str
    source_doc: str = ""


@dataclass
class KMetrics:
    k: int
    recall: float
    ndcg: float


@dataclass
class EvalResult:
    mrr: float
    per_k: list[KMetrics]
    latency_p50_ms: float
    latency_p95_ms: float
    n_samples: int
    n_hits: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mrr": round(self.mrr, 4),
            "per_k": [{"k": m.k, "recall": round(m.recall, 4), "ndcg": round(m.ndcg, 4)}
                      for m in self.per_k],
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "n_samples": self.n_samples,
            "n_hits": self.n_hits,
        }


# ── golden 片段命中判定 ────────────────────────────────────

def normalize(text: str) -> str:
    """去空白归一化, 用于片段包含判定"""
    return re.sub(r"\s+", "", text).lower()


def _longest_common_substr_ratio(a: str, b: str) -> float:
    """最长公共子串占 golden 片段的比例 (容错分块边界)"""
    from difflib import SequenceMatcher
    if not a:
        return 0.0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return match.size / len(a)


def is_hit(golden_fragment: str, content: str) -> bool:
    """判定检索到的 chunk 是否命中 golden 片段"""
    g, c = normalize(golden_fragment), normalize(content)
    if not g:
        return False
    if g in c:
        return True
    # 模糊兜底: 分块重切导致片段轻微变化, 公共子串占比 ≥ 60% 也算命中
    return _longest_common_substr_ratio(g, c) >= 0.6


# ── 数据集加载 ─────────────────────────────────────────────

def load_dataset(path: str | Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    seen_questions: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                question = str(obj.get("question", "")).strip()
                golden = str(obj.get("golden_fragment", obj.get("golden_chunk_content", ""))).strip()
                if not question or not golden:
                    print(f"[warn] 第 {line_no} 行缺少 question/golden_fragment, 跳过")
                    continue
                if question in seen_questions:
                    print(f"[warn] 第 {line_no} 行问题重复, 跳过")
                    continue
                seen_questions.add(question)
                samples.append(EvalSample(
                    question=question,
                    golden_fragment=golden,
                    source_doc=str(obj.get("source_doc", "")).strip(),
                ))
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {line_no} 行不是合法 JSON, 跳过: {e}")
    return samples


# ── 核心评测 ───────────────────────────────────────────────

async def evaluate(
    search_fn,
    dataset: list[EvalSample],
    top_k_list: list[int] | tuple[int, ...] = (5, 10),
    warmup_queries: int = 2,
) -> EvalResult:
    """
    对 search_fn 执行检索质量评测。

    search_fn: async (query: str, top_k: int) -> list[tuple[dict, float]]
               返回 ({"content": ..., ...}, score) 列表, 已按分数降序。
    """
    if not dataset:
        raise ValueError("数据集为空")

    max_k = max(top_k_list)
    reciprocal_ranks: list[float] = []
    dcg_sums: dict[int, float] = {k: 0.0 for k in top_k_list}
    hit_counts: dict[int, int] = {k: 0 for k in top_k_list}
    latencies_ms: list[float] = []

    # 排除模型/数据库首次初始化对检索延迟的污染。
    for sample in dataset[: max(0, warmup_queries)]:
        await search_fn(sample.question, max_k)

    for sample in dataset:
        t0 = time.perf_counter()
        results = await search_fn(sample.question, max_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        # 第一个命中的排名 (1-based), 0 表示未命中
        first_rank = 0
        for rank, (doc, _score) in enumerate(results, start=1):
            result_source = Path(str(doc.get("source", ""))).name
            source_matches = not sample.source_doc or result_source == Path(sample.source_doc).name
            if source_matches and is_hit(sample.golden_fragment, doc.get("content", "")):
                first_rank = rank
                break

        if first_rank:
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

        for k in top_k_list:
            if 0 < first_rank <= k:
                hit_counts[k] += 1
                # 单 golden 简化 DCG: 命中位置处 rel=1
                dcg_sums[k] += 1.0 / math.log2(first_rank + 1)

    n = len(dataset)
    return EvalResult(
        mrr=sum(reciprocal_ranks) / n,
        per_k=[KMetrics(
            k=k,
            recall=hit_counts[k] / n,
            ndcg=dcg_sums[k] / n,  # 单 golden 场景 IDCG=1, NDCG=DCG
        ) for k in top_k_list],
        latency_p50_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
        latency_p95_ms=_percentile(latencies_ms, 95),
        n_samples=n,
        n_hits={f"recall@{k}": hit_counts[k] for k in top_k_list},
    )


def _percentile(values: list[float], pct: float) -> float:
    """线性插值百分位数"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (pct / 100.0) * (len(ordered) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


# ── 报告输出 ───────────────────────────────────────────────

def print_report(title: str, result: EvalResult) -> None:
    print(f"\n=== {title} ===")
    print(f"样本数: {result.n_samples}")
    print(f"MRR:     {result.mrr:.4f}")
    for m in result.per_k:
        hits = result.n_hits.get(f"recall@{m.k}", 0)
        print(f"Recall@{m.k}: {m.recall:.4f}  ({hits}/{result.n_samples} 命中)  NDCG@{m.k}: {m.ndcg:.4f}")
    print(f"延迟:    P50={result.latency_p50_ms:.1f}ms  P95={result.latency_p95_ms:.1f}ms")


# ── CLI ────────────────────────────────────────────────────

async def _run_cli(args) -> None:
    dataset = load_dataset(args.dataset)
    print(f"加载数据集: {len(dataset)} 条 (来自 {args.dataset})")
    if args.search_impl == "mock":
        # mock 模式: 每个问题返回 golden 片段排第一, 用于验证指标计算
        async def mock_search(query: str, top_k: int):
            sample = next(s for s in dataset if s.question == query)
            return [({"content": sample.golden_fragment, "source": sample.source_doc}, 0.99)] + \
                   [({"content": f"无关内容{i}", "source": "mock"}, 0.5 - i * 0.01) for i in range(top_k - 1)]

        result = await evaluate(mock_search, dataset, args.top_k)
        print_report("Mock 自检 (应 Recall≈1.0, MRR≈1.0)", result)
        return

    # 真实评测: 通过注册表注入 search_fn (由 baseline_vs_modern.py 注册)
    from benchmarks import eval_retrieval
    if not hasattr(eval_retrieval, "_registered_search_fn"):
        raise SystemExit("请先注册 search_fn (eval_retrieval._registered_search_fn = fn), "
                         "或直接以编程方式调用 evaluate()。"
                         "完整对比流程请运行: python benchmarks/baseline_vs_modern.py")
    result = await evaluate(eval_retrieval._registered_search_fn, dataset, args.top_k)
    print_report("检索评测结果", result)
    if args.out:
        Path(args.out).write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测 (Recall@K / MRR / NDCG@K / 延迟)")
    parser.add_argument("--dataset", required=True, help="评测数据集 JSONL 路径")
    parser.add_argument("--top-k", default="5,10", help="逗号分隔的 K 值列表 (默认 5,10)")
    parser.add_argument("--search-impl", default="registered", choices=["registered", "mock"],
                        help="registered: 使用注入的 search_fn; mock: 指标计算自检")
    parser.add_argument("--out", default="", help="结果保存路径 (JSON)")
    args = parser.parse_args()
    args.top_k = [int(x) for x in args.top_k.split(",")]
    asyncio.run(_run_cli(args))


if __name__ == "__main__":
    main()
