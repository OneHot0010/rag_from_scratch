"""阶段2/3 · 模块化 RAG 交互式入口。

用法:
    python main.py                          # 用 .env 中 DOC_PATH 与默认切分策略
    python main.py data.csv a.xlsx          # 指定文件(自动按扩展名分派)
    python main.py doc.md --strategy markdown   # 指定切分策略
    python main.py --strategy semantic      # 语义切分(需 API Key)

相比阶段1 的 demo.py,核心逻辑已拆分到 ragcore/ 各模块:
数据接入抽象为 DataLoader(支持 txt/csv/xlsx),切分抽象为可插拔策略族
(fixed/recursive/sentence/paragraph/markdown/semantic,见 ragcore/splitters)。
"""
from __future__ import annotations

import argparse

from ragcore.config import settings
from ragcore.pipeline import RAGPipeline
from ragcore.splitters import available_strategies


def main() -> None:
    ap = argparse.ArgumentParser(description="模块化 RAG 问答")
    ap.add_argument("paths", nargs="*", help="待索引文件(默认用 .env 的 DOC_PATH)")
    ap.add_argument(
        "--strategy",
        choices=available_strategies(),
        default=None,
        help=f"切分策略(默认 {settings.split_strategy})",
    )
    args = ap.parse_args()

    paths = args.paths or [settings.doc_path]
    pipeline = RAGPipeline(split_strategy=args.strategy)
    print(f"支持的格式: {', '.join(pipeline.registry.supported_extensions())}")
    print(f"切分策略: {pipeline.strategy}")
    print(f"正在加载并向量化: {', '.join(paths)} ...")
    n = pipeline.build_index(paths)
    print(f"索引构建完成,共 {n} 个 chunk。输入问题开始问答(输入 q 退出)\n")

    while True:
        try:
            query = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        answer, hits = pipeline.ask(query)
        print("\n--- 检索到的 chunk ---")
        for i, (doc, score) in enumerate(hits):
            src = doc.metadata.get("source", "?")
            print(f"[{i + 1}] (相似度 {score:.3f}) [{src}] {doc.content[:80]} ...")
        print(f"\n答案: {answer}\n")


if __name__ == "__main__":
    main()
