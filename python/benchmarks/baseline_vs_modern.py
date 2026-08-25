"""
Baseline vs Modern 检索对比评测

Baseline: 固定 512 字符分块 + Chroma 本地 + 纯向量检索 (无 reranker)
Modern:   语义/结构分块 + 向量库 (chroma|qdrant) + BGE-Reranker 精排

同一组文档、同一组问题, 分别评测后输出对比表:
  Recall@K / MRR / NDCG@K / 延迟 P50/P95

golden set 来源 (二选一):
  --dataset qa_pairs.jsonl   已有问答对 (generate_qa_pairs.py 产出)
  --docs 目录                 无数据集时自动离线合成 (核心句 + 锚词问题)

用法:
  python benchmarks/baseline_vs_modern.py --docs benchmarks/sample_docs --top-k 5
  python benchmarks/baseline_vs_modern.py --docs benchmarks/sample_docs --dataset qa_pairs.jsonl
  python benchmarks/baseline_vs_modern.py --vector-store qdrant      # modern 后端切换
  python benchmarks/baseline_vs_modern.py --no-reranker              # 跳过 BGE 精排 (模型未就绪时)

输出对比示例:
  ┌─────────────┬────────────┬────────────┬──────────┐
  │ 指标        │ Baseline   │ Modern     │ 提升     │
  ├─────────────┼────────────┼────────────┼──────────┤
  │ Recall@5    │ 0.62       │ 0.84       │ +35.5%   │
  │ MRR         │ 0.48       │ 0.71       │ +47.9%   │
  │ P95 延迟    │ 2.8s       │ 0.9s       │ -67.9%   │
  └─────────────┴────────────┴────────────┴──────────┘
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_retrieval import EvalSample, evaluate, load_dataset  # noqa: E402
from benchmarks.generate_qa_pairs import (  # noqa: E402
    _iter_doc_files,
    fixed_chunk,
    generate_offline,
)


@dataclass
class ComparisonRow:
    metric: str
    baseline: str
    modern: str
    improvement: str


# ── Baseline: 固定分块 + Chroma 原生 + 纯向量 ──────────────

class BaselineIndex:
    """模拟旧系统: chromadb 原生 API, 固定分块, 无 reranker"""

    def __init__(self, embeddings, persist_dir: str) -> None:
        self.embeddings = embeddings
        self._client = None
        self._collection = None
        self.persist_dir = persist_dir

    def init(self) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(name="baseline_chunks")

    def close(self) -> None:
        """释放底层文件句柄 (Windows 上 chromadb 会锁住 SQLite 文件)"""
        import gc
        client = self._client
        self._collection = None
        self._client = None
        system = getattr(client, "_system", None)
        stop = getattr(system, "stop", None)
        if callable(stop):
            stop()
        gc.collect()

    def add_documents(self, doc_texts: dict[str, str], chunk_size: int = 512) -> None:
        """固定字符分块入库 (旧系统行为)"""
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        all_texts: list[str] = []
        for name, text in doc_texts.items():
            for i, piece in enumerate(fixed_chunk(text, chunk_size=chunk_size)):
                ids.append(f"{name}#{i}")
                docs.append(piece)
                metas.append({"source_doc": name})
                all_texts.append(piece)
        if not all_texts:
            return
        vectors = self.embeddings.embed_documents(all_texts)
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

    async def search_fn(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        q_vec = await self.embeddings.aembed_query(query)
        res = self._collection.query(
            query_embeddings=[q_vec], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[dict, float]] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(({"content": doc, "source": meta.get("source_doc", "")}, 1.0 - dist))
        return out


# ── Modern: 语义分块 + VectorStoreService + BGE-Reranker ───

class ModernIndex:
    """现代管线: 复用项目服务层 (解析 → 语义分块 → 向量库 → reranker)"""

    def __init__(self, embeddings, backend: str, persist_dir: str, use_reranker: bool) -> None:
        from config import settings
        self.embeddings = embeddings
        self.backend = backend
        self.persist_dir = persist_dir
        self.use_reranker = use_reranker
        self.reranker = None
        self.reranker_top_n = settings.reranker_top_n
        self.store = None

    async def init(self) -> None:
        from config import settings
        from services.vector_store import VectorStoreService

        settings.vector_store_type = self.backend
        if self.backend == "qdrant":
            settings.qdrant_url = ""
            settings.qdrant_path = self.persist_dir
        else:
            settings.chroma_path = self.persist_dir

        self.store = VectorStoreService(embeddings=self.embeddings)
        await self.store.init()

        if self.use_reranker:
            from services.reranker import RerankerService
            self.reranker = RerankerService()
            if not self.reranker.available:
                print("[warn] BGE-Reranker 模型不可用, 跳过精排")
                self.reranker = None

    async def close(self) -> None:
        """释放底层存储文件句柄 (Windows 文件锁)"""
        import gc
        if self.store is not None:
            await self.store.close()
            self.store = None
        self.reranker = None
        gc.collect()

    async def add_documents(self, doc_texts: dict[str, str]) -> None:
        """解析 + 语义分块入库 (现代管线)"""
        from schema import DocType
        from services.document_parser import ParseBlock, UnifiedDocumentParser
        from utils.chunker import DocumentChunker

        parser = UnifiedDocumentParser()
        chunker = DocumentChunker(embeddings=self.embeddings, semantic_enabled=True)
        for i, (name, text) in enumerate(doc_texts.items()):
            blocks = parser._markdown_to_blocks(text)
            if not blocks:
                blocks = [ParseBlock(kind="text", content=text)]
            chunks = chunker.chunk(blocks, doc_id=f"doc{i}", doc_type=DocType.MARKDOWN, source=name)
            await self.store.add_chunks(chunks)

    async def search_fn(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        candidates = await self.store.search(query, top_k=max(top_k, self.reranker_top_n))
        if self.reranker is not None and len(candidates) > 1:
            scores = await self.reranker.arerank(query, [d["content"] for d, _ in candidates])
            if scores is not None:
                ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
                candidates = [(d, s) for (d, _), s in ranked]
        return candidates[:top_k]


# ── 对比执行 ───────────────────────────────────────────────

def _load_golden_set(
    dataset_path: str | None,
    doc_texts: dict[str, str],
    max_per_chunk: int,
) -> tuple[list[EvalSample], str]:
    if dataset_path:
        samples = load_dataset(dataset_path)
        print(f"golden set: {len(samples)} 条 (人工/外部数据集: {dataset_path})")
        return samples, "curated_or_external"
    # 离线合成: 对现代分块结果抽取核心句 (与生成脚本同一套逻辑)
    chunks = [{"content": piece, "source_doc": name}
              for name, text in doc_texts.items() for piece in fixed_chunk(text)]
    rows = generate_offline(chunks, max_per_chunk=max_per_chunk)
    samples = [EvalSample(question=r["question"], golden_fragment=r["golden_fragment"],
                          source_doc=r["source_doc"]) for r in rows]
    print(f"golden set: {len(samples)} 条 (合成演示集, 每 chunk ≤{max_per_chunk} 对; 不应用于简历指标)")
    return samples, "synthetic_demo"


def _sha256_texts(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fmt_pct_change(base: float, modern: float) -> str:
    if base == 0:
        return "N/A"
    return f"{(modern - base) / base * 100:+.1f}%"


def _build_table(rows: list[ComparisonRow]) -> str:
    width = 13
    sep = "─" * width
    lines = [
        f"┌{'┬'.join(sep for _ in range(4))}┐",
        f"│{'指标'.center(width)}│{'Baseline'.center(width)}│{'Modern'.center(width)}│{'提升'.center(width)}│",
        f"├{'┼'.join(sep for _ in range(4))}┤",
    ]
    for r in rows:
        lines.append(
            f"│{r.metric.center(width)}│{r.baseline.center(width)}│"
            f"{r.modern.center(width)}│{r.improvement.center(width)}│"
        )
    lines.append(f"└{'┴'.join(sep for _ in range(4))}┘")
    return "\n".join(lines)


async def _run(args) -> None:
    from utils.embeddings import create_embeddings

    embeddings = create_embeddings()

    # 1. 加载文档
    files = _iter_doc_files(args.docs)
    if not files:
        raise SystemExit(f"目录 {args.docs} 下没有 md/txt 文档")
    doc_texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    print(f"评测文档: {len(files)} 个 ({', '.join(doc_texts)})")

    samples, dataset_kind = _load_golden_set(args.dataset, doc_texts, args.max_per_chunk)

    results = {}

    # 2. Baseline
    with tempfile.TemporaryDirectory(prefix="bench_baseline_", ignore_cleanup_errors=True) as tmp:
        baseline = BaselineIndex(embeddings, persist_dir=tmp)
        baseline.init()
        baseline.add_documents(doc_texts, chunk_size=512)
        print("\n[Baseline] 固定512分块 + Chroma + 纯向量检索 评测中...")
        results["baseline"] = await evaluate(baseline.search_fn, samples, args.top_k)
        baseline.close()

    # 3. Modern
    with tempfile.TemporaryDirectory(prefix="bench_modern_", ignore_cleanup_errors=True) as tmp:
        modern = ModernIndex(embeddings, backend=args.vector_store,
                             persist_dir=tmp, use_reranker=not args.no_reranker)
        await modern.init()
        await modern.add_documents(doc_texts)
        label = f"语义分块 + {args.vector_store} + {'BGE-Reranker' if modern.reranker else '无精排'}"
        reranker_enabled = modern.reranker is not None
        print(f"\n[Modern] {label} 评测中...")
        results["modern"] = await evaluate(modern.search_fn, samples, args.top_k)
        await modern.close()

    # 4. 对比表
    base, mod = results["baseline"], results["modern"]
    rows: list[ComparisonRow] = [
        ComparisonRow("MRR", f"{base.mrr:.4f}", f"{mod.mrr:.4f}", _fmt_pct_change(base.mrr, mod.mrr)),
    ]
    for bm, mm in zip(base.per_k, mod.per_k):
        rows.append(ComparisonRow(
            f"Recall@{bm.k}", f"{bm.recall:.4f}", f"{mm.recall:.4f}",
            _fmt_pct_change(bm.recall, mm.recall),
        ))
        rows.append(ComparisonRow(
            f"NDCG@{bm.k}", f"{bm.ndcg:.4f}", f"{mm.ndcg:.4f}",
            _fmt_pct_change(bm.ndcg, mm.ndcg),
        ))
    rows.append(ComparisonRow(
        "P50 延迟(ms)", f"{base.latency_p50_ms:.0f}", f"{mod.latency_p50_ms:.0f}",
        _fmt_pct_change(base.latency_p50_ms, mod.latency_p50_ms),
    ))
    rows.append(ComparisonRow(
        "P95 延迟(ms)", f"{base.latency_p95_ms:.0f}", f"{mod.latency_p95_ms:.0f}",
        _fmt_pct_change(base.latency_p95_ms, mod.latency_p95_ms),
    ))

    print("\n对比结果:")
    print(_build_table(rows))
    if reranker_enabled:
        print("\n注: Modern 延迟包含本机 BGE-Reranker 推理；实际延迟取决于模型、硬件和批大小。")
    if dataset_kind == "synthetic_demo":
        print("警告: 当前结果来自自动合成演示集，只适合回归测试，不应作为业务准确率写入简历。")

    # 5. 保存结果 (供简历/面试引用)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "docs": list(doc_texts),
                "docs_sha256": _sha256_texts([f"{name}\n{text}" for name, text in doc_texts.items()]),
                "dataset": args.dataset or "auto-generated",
                "dataset_kind": dataset_kind,
                "dataset_sha256": _sha256_texts([
                    f"{sample.question}\n{sample.golden_fragment}\n{sample.source_doc}"
                    for sample in samples
                ]),
                "n_samples": len(samples),
                "top_k": args.top_k,
                "vector_store": args.vector_store,
                "reranker": reranker_enabled,
                "python": platform.python_version(),
            },
            "baseline": results["baseline"].as_dict(),
            "modern": results["modern"].as_dict(),
            "rows": [r.__dict__ for r in rows],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs Modern 检索对比评测")
    parser.add_argument("--docs", required=True, help="评测文档目录 (md/txt)")
    parser.add_argument("--dataset", default="", help="已有 golden set JSONL (缺省则离线合成)")
    parser.add_argument("--top-k", default="5,10", help="逗号分隔的 K 值 (默认 5,10)")
    parser.add_argument("--vector-store", default="chroma", choices=["chroma", "qdrant"],
                        help="Modern 后端 (默认 chroma; qdrant 走嵌入式本地模式)")
    parser.add_argument("--no-reranker", action="store_true", help="跳过 BGE-Reranker 精排")
    parser.add_argument("--max-per-chunk", type=int, default=2, help="离线合成时每 chunk 最多问答对数")
    parser.add_argument("--out", default="benchmarks/results/baseline_vs_modern.json", help="结果保存路径")
    args = parser.parse_args()
    args.top_k = [int(x) for x in args.top_k.split(",")]
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
