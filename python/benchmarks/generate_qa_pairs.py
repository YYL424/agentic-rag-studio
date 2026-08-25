"""
自动从文档 chunk 生成问答对, 作为检索评测的 golden set。

两种模式:
  1. LLM 模式 (默认): 对每个 chunk 调用 LLM 生成 1-2 个问答对,
     问题应能从文本中直接回答。返回 JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}
  2. 离线模式 (--no-llm): 从 chunk 中抽取信息密度高的核心句作为 golden 片段,
     合成检索型问题。无需 LLM key, 结果可直接用于 eval_retrieval.py。

输出 JSONL, 每行:
  {"question": "...", "golden_fragment": "...", "source_doc": "...", "answer": "..."}

用法:
  python benchmarks/generate_qa_pairs.py --docs benchmarks/sample_docs --out qa_pairs.jsonl
  python benchmarks/generate_qa_pairs.py --docs path --out qa_pairs.jsonl --no-llm --max-per-chunk 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 句子切分: 中文句号/问号/叹号/分号 + 换行
_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")

# 信息密度加权: 含数字/中文逗号/书名号的句子更可能承载具体事实
_DENSITY_RE = re.compile(r"[0-9０-９,，、《》%％]")

QA_PROMPT = """\
你是一个 RAG 检索评测数据标注员。请基于以下文本生成 1-2 个问答对。
要求:
1. 问题应能从文本中直接回答 (事实型问题)
2. 问题尽量具体, 包含文本中的实体或数字, 便于检索召回
3. 返回 JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}

文本:
{chunk}
"""


def _iter_doc_files(docs_dir: str | Path) -> list[Path]:
    docs = Path(docs_dir)
    files = sorted(
        p for p in docs.rglob("*")
        if p.suffix.lower() in (".md", ".txt")
    )
    return files


def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """固定字符分块 (与 chunker 的 fallback 行为一致)"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_core_sentences(chunk: str, max_len: int = 80, min_len: int = 12) -> list[str]:
    """从 chunk 中抽取信息密度高的核心句 (作 golden 片段候选)"""
    sentences = [s.strip() for s in _SENT_RE.findall(chunk) if s.strip()]
    scored: list[tuple[float, str]] = []
    for sent in sentences:
        if not (min_len <= len(sent) <= max_len):
            continue
        score = 0.0
        if _DENSITY_RE.search(sent):
            score += 1.0
        if "，" in sent:
            score += 0.5  # 复合句通常承载更多信息
        scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


def synthesize_question(sentence: str) -> str:
    """离线模式: 由核心句合成检索型问题。

    锚词取句子中段 8-12 字 (而非开头), 降低字面重合度,
    更接近真实用户的查询形态, 让评测有区分度。
    """
    if len(sentence) <= 24:
        anchor = sentence
    else:
        start = len(sentence) // 3
        anchor = sentence[start : start + 10]
    return f'文档中提到"{anchor}……"，具体内容是什么？'


async def generate_with_llm(chunks: list[dict]) -> list[dict]:
    """LLM 模式: 每个 chunk 生成 1-2 个自然问答对"""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from config import settings

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    rows: list[dict] = []
    for ch in chunks:
        resp = await llm.ainvoke([HumanMessage(content=QA_PROMPT.format(chunk=ch["content"]))])
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            pairs = json.loads(cleaned).get("qa_pairs", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        for pair in pairs:
            q, a = pair.get("question", ""), pair.get("answer", "")
            if not q or not a:
                continue
            # golden 片段: 优先用答案原文, 过长则截取前 80 字
            golden = a if len(a) <= 80 else a[:80]
            rows.append({
                "question": q,
                "golden_fragment": golden,
                "source_doc": ch["source_doc"],
                "answer": a,
            })
    return rows


def generate_offline(chunks: list[dict], max_per_chunk: int) -> list[dict]:
    """离线模式: 核心句 + 合成问题 (无需 LLM)"""
    rows: list[dict] = []
    for ch in chunks:
        for sent in extract_core_sentences(ch["content"])[:max_per_chunk]:
            rows.append({
                "question": synthesize_question(sent),
                "golden_fragment": sent,
                "source_doc": ch["source_doc"],
                "answer": sent,
            })
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description="从文档自动生成 RAG 评测问答对")
    parser.add_argument("--docs", required=True, help="文档目录 (md/txt)")
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    parser.add_argument("--no-llm", action="store_true", help="离线模式: 不调 LLM, 合成问题")
    parser.add_argument("--max-per-chunk", type=int, default=2, help="每 chunk 最多生成问答对数 (默认 2)")
    parser.add_argument("--chunk-size", type=int, default=512, help="固定分块大小 (默认 512)")
    args = parser.parse_args()

    files = _iter_doc_files(args.docs)
    if not files:
        raise SystemExit(f"目录 {args.docs} 下没有 md/txt 文档")

    chunks: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for piece in fixed_chunk(text, chunk_size=args.chunk_size):
            chunks.append({"content": piece, "source_doc": path.name})

    print(f"文档 {len(files)} 个 → {len(chunks)} 个 chunk")

    if args.no_llm:
        rows = generate_offline(chunks, args.max_per_chunk)
    else:
        try:
            rows = await generate_with_llm(chunks)
        except Exception as e:
            print(f"[warn] LLM 生成失败 ({e}), 降级为离线合成模式")
            rows = generate_offline(chunks, args.max_per_chunk)

    if not rows:
        raise SystemExit("未能生成任何问答对, 请检查文档内容")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"生成 {len(rows)} 个问答对 → {out}")


if __name__ == "__main__":
    asyncio.run(main())
