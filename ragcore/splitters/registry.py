"""切分策略注册表:按策略名分派与构建。

与 loaders 的 registry 对称:新增策略的唯一改动点是在这里登记一个工厂。
上层(pipeline / 实验脚本)只按名字取策略,无需感知具体实现(开闭原则)。

语义切分(semantic)依赖 embedder,故按需惰性构建:仅当请求该策略时才
要求传入 embedder。
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .base import Splitter
from .fixed import FixedSizeSplitter
from .recursive import RecursiveCharacterSplitter
from .sentence import SentenceSplitter, ParagraphSplitter
from .markdown import MarkdownSplitter
from .semantic import SemanticSplitter

# 不依赖外部资源、可离线构建的策略工厂:name -> (chunk_size, overlap) -> Splitter
_FACTORIES: Dict[str, Callable[[int, int], Splitter]] = {
    FixedSizeSplitter.name: lambda cs, ov: FixedSizeSplitter(cs, ov),
    RecursiveCharacterSplitter.name: lambda cs, ov: RecursiveCharacterSplitter(cs, ov),
    SentenceSplitter.name: lambda cs, ov: SentenceSplitter(cs, ov),
    ParagraphSplitter.name: lambda cs, ov: ParagraphSplitter(cs, ov),
    MarkdownSplitter.name: lambda cs, ov: MarkdownSplitter(cs, ov),
}

#: 需要 embedder、无法离线构建的策略名(供实验脚本判断是否跳过)。
EMBEDDING_STRATEGIES = (SemanticSplitter.name,)


def available_strategies(include_semantic: bool = True) -> List[str]:
    """列出可用策略名。"""
    names = list(_FACTORIES)
    if include_semantic:
        names += list(EMBEDDING_STRATEGIES)
    return names


def build_splitter(
    strategy: str,
    chunk_size: int = 300,
    overlap: int = 50,
    embedder=None,
    **kwargs,
) -> Splitter:
    """按名字构建切分策略。

    Args:
        strategy: 策略名,见 available_strategies()。
        chunk_size / overlap: 通用切分参数。
        embedder: 仅 semantic 策略需要。
        **kwargs: 透传给 semantic 等策略的额外参数。
    """
    strategy = (strategy or "").lower()
    if strategy in _FACTORIES:
        return _FACTORIES[strategy](chunk_size, overlap)
    if strategy == SemanticSplitter.name:
        if embedder is None:
            raise ValueError("semantic 策略需要传入 embedder")
        return SemanticSplitter(embedder=embedder, **kwargs)
    raise ValueError(
        f"未知切分策略: {strategy!r}。可用: {', '.join(available_strategies())}"
    )
