"""切分策略层:阶段3 · 把 Chunking 抽象为可插拔、可对比的策略族。

已实现策略:
    - fixed      固定长度(带 overlap)——最简兜底
    - recursive  递归字符切分——通用默认,尽量保住语义边界
    - sentence   按句切分 + 贪心合并
    - paragraph  按段切分,超长段落回退句子切分
    - markdown   结构感知,按标题层级切并注入标题路径元数据
    - semantic   语义切分(需 embedder,按需启用)
"""
from .base import Splitter
from .fixed import FixedSizeSplitter
from .recursive import RecursiveCharacterSplitter
from .sentence import SentenceSplitter, ParagraphSplitter
from .markdown import MarkdownSplitter
from .semantic import SemanticSplitter
from .registry import (
    build_splitter,
    available_strategies,
    EMBEDDING_STRATEGIES,
)

__all__ = [
    "Splitter",
    "FixedSizeSplitter",
    "RecursiveCharacterSplitter",
    "SentenceSplitter",
    "ParagraphSplitter",
    "MarkdownSplitter",
    "SemanticSplitter",
    "build_splitter",
    "available_strategies",
    "EMBEDDING_STRATEGIES",
]
