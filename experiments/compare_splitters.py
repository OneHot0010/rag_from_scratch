"""阶段3 · 切分策略对照实验。

在同一份文档上跑遍所有切分策略,输出可横向对比的指标表:
    chunk 数 / 长度(min·avg·max)/ 长度标准差(越小越均匀)/ 样例。

离线策略(fixed·recursive·sentence·paragraph·markdown)无需 API 即可运行;
语义策略(semantic)依赖 Embedding,仅当配置了 API Key 时才纳入对比。

用法:
    python experiments/compare_splitters.py                # 用默认 sample.txt
    python experiments/compare_splitters.py path/to/doc    # 指定文档
    python experiments/compare_splitters.py --size 200 --overlap 40
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# 允许直接以脚本方式运行(把项目根加入 import 路径)。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragcore.models import Document
from ragcore.splitters import (
    build_splitter,
    available_strategies,
    EMBEDDING_STRATEGIES,
)


def _load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"文档不存在: {path}")
    return p.read_text(encoding="utf-8")


def _stats(chunks):
    """把一批 chunk 折算成可比较的数值指标。"""
    lens = [len(c.content) for c in chunks] or [0]
    return {
        "count": len(chunks),
        "min": min(lens),
        "avg": sum(lens) // len(lens),
        "max": max(lens),
        "std": round(statistics.pstdev(lens), 1) if len(lens) > 1 else 0.0,
    }


def _fmt_table(rows):
    """渲染定宽对齐的对比表(纯文本,便于终端查看)。"""
    header = ["策略", "chunk数", "min", "avg", "max", "长度std", "溯源元数据"]
    widths = [12, 8, 5, 5, 5, 8, 20]
    line = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "-+-".join("-" * w for w in widths)
    out = [line, sep]
    for r in rows:
        cells = [
            r["name"].ljust(widths[0]),
            str(r["count"]).ljust(widths[1]),
            str(r["min"]).ljust(widths[2]),
            str(r["avg"]).ljust(widths[3]),
            str(r["max"]).ljust(widths[4]),
            str(r["std"]).ljust(widths[5]),
            r["meta"].ljust(widths[6]),
        ]
        out.append(" | ".join(cells))
    return "\n".join(out)


def _maybe_embedder():
    """有 API Key 才构建 embedder,否则返回 None(跳过 semantic)。"""
    try:
        from ragcore.config import settings
        if not settings.api_key:
            return None
        from volcenginesdkarkruntime import Ark
        from ragcore.embedder import Embedder
        client = Ark(api_key=settings.api_key, base_url=settings.base_url)
        return Embedder(client=client)
    except Exception as e:  # noqa: BLE001 - 实验脚本,容错跳过
        print(f"[提示] 无法初始化 embedder,跳过 semantic:{e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="切分策略对照实验")
    ap.add_argument("doc", nargs="?", default="sample.txt", help="待切分文档路径")
    ap.add_argument("--size", type=int, default=200, help="chunk_size(默认 200)")
    ap.add_argument("--overlap", type=int, default=40, help="overlap(默认 40)")
    ap.add_argument("--samples", type=int, default=1, help="每策略展示的样例 chunk 数")
    args = ap.parse_args()

    text = _load_text(args.doc)
    doc = Document(content=text, metadata={"source": Path(args.doc).name})
    print(f"文档: {args.doc}  总字符数: {len(text)}")
    print(f"参数: chunk_size={args.size}  overlap={args.overlap}\n")

    embedder = _maybe_embedder()
    rows = []
    samples = {}
    for name in available_strategies():
        if name in EMBEDDING_STRATEGIES and embedder is None:
            print(f"[跳过] {name}:未配置 API Key")
            continue
        sp = build_splitter(name, chunk_size=args.size, overlap=args.overlap,
                            embedder=embedder)
        chunks = sp.split_documents([doc])
        st = _stats(chunks)
        meta_keys = set()
        for c in chunks:
            meta_keys |= set(c.metadata.keys())
        extra = sorted(meta_keys - {"source", "chunk", "splitter"})
        rows.append({
            "name": name,
            "meta": ",".join(extra) if extra else "-",
            **st,
        })
        samples[name] = chunks[: args.samples]

    print(_fmt_table(rows))
    print("\n=== 样例 chunk ===")
    for name, chs in samples.items():
        print(f"\n[{name}]")
        for c in chs:
            hp = c.metadata.get("heading_path", "")
            tag = f"  ({hp})" if hp else ""
            body = c.content.replace("\n", " ")
            print(f"  #{c.metadata['chunk']}{tag} {body[:70]}")

    print("\n提示:参数建议与策略选型见 docs/切分策略.md")


if __name__ == "__main__":
    main()
