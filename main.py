"""阶段2 · 模块化 RAG 交互式入口。

用法:
    python main.py                    # 用 .env 中 DOC_PATH 指定的文件
    python main.py data.csv a.xlsx    # 指定一个或多个文件(自动按扩展名分派)

相比阶段1 的 demo.py,核心逻辑已拆分到 ragcore/ 各模块,
数据接入抽象为 DataLoader 接口,当前支持 txt / csv / xlsx。
"""
from __future__ import annotations

import sys

from ragcore.config import settings
from ragcore.pipeline import RAGPipeline


def main() -> None:
    paths = sys.argv[1:] or [settings.doc_path]
    pipeline = RAGPipeline()
    print(f"支持的格式: {', '.join(pipeline.registry.supported_extensions())}")
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
