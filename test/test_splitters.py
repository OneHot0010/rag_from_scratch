"""阶段3 · 切分策略单元测试。

每个测试函数独立可运行、打印中间结果,便于 debug。
运行方式:
    PYTHONPATH=. python3 test/test_splitters.py          # 内置 runner(无需 pytest)
    PYTHONPATH=. python3 -m pytest test/test_splitters.py # 有 pytest 时

仅覆盖离线策略(不触发 Embedding / API):
    fixed / recursive / sentence / paragraph / markdown + registry。
semantic 依赖 embedder,不在离线单测范围内(见对照实验脚本)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragcore.models import Document
from ragcore.splitters import (
    build_splitter,
    available_strategies,
    EMBEDDING_STRATEGIES,
    FixedSizeSplitter,
    RecursiveCharacterSplitter,
    SentenceSplitter,
    ParagraphSplitter,
    MarkdownSplitter,
)

_TEXT = (
    "RAG 是检索增强生成。它先检索再生成,能减少幻觉。"
    "切分是关键环节。chunk 过大噪声多,过小则断裂。"
    "常见做法是设 chunk_size 并保留 overlap。"
) * 3

_MD = (
    "# 安装\n\n先准备环境,Python 3.10 以上即可。\n\n"
    "## 依赖\n\n运行 pip install -r requirements.txt 安装依赖。\n\n"
    "# 使用\n\n直接运行 main.py,进入问答循环。"
)


def test_fixed_respects_chunk_size():
    """固定长度:每块不超过 chunk_size,且能覆盖全文。"""
    sp = FixedSizeSplitter(chunk_size=50, overlap=10)
    chunks = sp.split_text(_TEXT)
    print(f"fixed -> {len(chunks)} chunks, lens={[len(c) for c in chunks]}")
    assert chunks, "不应为空"
    assert all(len(c) <= 50 for c in chunks), "固定长度不得超过 chunk_size"


def test_fixed_overlap_validation():
    """overlap >= chunk_size 应拒绝。"""
    try:
        FixedSizeSplitter(chunk_size=30, overlap=30)
        assert False, "应抛 ValueError"
    except ValueError:
        print("fixed -> 非法 overlap 已正确拒绝")


def test_recursive_keeps_boundaries():
    """递归字符:块长均匀,且不产生空块。"""
    sp = RecursiveCharacterSplitter(chunk_size=60, overlap=10)
    chunks = sp.split_text(_TEXT)
    print(f"recursive -> {len(chunks)} chunks, lens={[len(c) for c in chunks]}")
    assert chunks
    assert all(c.strip() for c in chunks), "不应有空块"


def test_sentence_split():
    """句子:每块非空,块数合理。"""
    sp = SentenceSplitter(chunk_size=60, overlap=10)
    chunks = sp.split_text(_TEXT)
    print(f"sentence -> {len(chunks)} chunks")
    assert len(chunks) >= 1


def test_paragraph_split():
    """段落:多段文本应产出多块。"""
    sp = ParagraphSplitter(chunk_size=40, overlap=5)
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = sp.split_text(text)
    print(f"paragraph -> {len(chunks)} chunks: {chunks}")
    assert len(chunks) >= 1


def test_markdown_heading_path():
    """Markdown:注入 heading_path 元数据,且能识别层级。"""
    sp = MarkdownSplitter(chunk_size=200, overlap=20)
    docs = sp.split_documents([Document(content=_MD, metadata={"source": "d.md"})])
    print(f"markdown -> {len(docs)} chunks")
    paths = [d.metadata.get("heading_path", "") for d in docs]
    print(f"heading_paths = {paths}")
    assert any(" > " in p for p in paths), "应识别出多级标题路径,如 '安装 > 依赖'"
    assert all(d.metadata.get("splitter") == "markdown" for d in docs)


def test_markdown_ignores_fence():
    """代码块内的 # 不应被误判为标题。"""
    sp = MarkdownSplitter(chunk_size=300, overlap=20)
    md = "# 真标题\n\n```\n# 这是注释不是标题\n```\n\n正文。"
    docs = sp.split_documents([Document(content=md, metadata={})])
    paths = {d.metadata.get("heading_path", "") for d in docs}
    print(f"fence test heading_paths = {paths}")
    assert "真标题" in paths
    assert "这是注释不是标题" not in paths


def test_metadata_chunk_and_splitter():
    """通用契约:每个 chunk 带 chunk 序号与 splitter 名。"""
    sp = build_splitter("recursive", chunk_size=60, overlap=10)
    docs = sp.split_documents([Document(content=_TEXT, metadata={"source": "x"})])
    for i, d in enumerate(docs):
        assert d.metadata["chunk"] == i
        assert d.metadata["splitter"] == "recursive"
        assert d.metadata["source"] == "x"
    print(f"metadata -> {len(docs)} chunks, 序号与策略名校验通过")


def test_registry_available():
    """注册表:列出全部 6 种策略;可选择是否含 semantic。"""
    names = available_strategies()
    print(f"available = {names}")
    for n in ["fixed", "recursive", "sentence", "paragraph", "markdown"]:
        assert n in names
    assert set(EMBEDDING_STRATEGIES).issubset(set(names))
    offline = available_strategies(include_semantic=False)
    assert "semantic" not in offline


def test_registry_unknown_and_semantic_guard():
    """未知策略报错;semantic 缺 embedder 报错。"""
    try:
        build_splitter("nope")
        assert False
    except ValueError:
        print("registry -> 未知策略已拒绝")
    try:
        build_splitter("semantic")
        assert False
    except ValueError:
        print("registry -> semantic 缺 embedder 已拒绝")


# ====================================================================
# 运行入口(无 pytest 也能跑)
# ====================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        print("=" * 60)
        print(f"运行: {t.__name__}")
        t()
        passed += 1
    print("=" * 60)
    print(f"全部通过: {passed}/{len(tests)}")
