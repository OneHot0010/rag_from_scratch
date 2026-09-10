"""固定长度切分(Fixed-size)。

按字符数硬切,带 overlap 防止关键信息被切分点截断。
最简单、最可控,但会切断语义,仅适合结构弱的纯文本兜底。

对应阶段1/2 的 `splitter.split_text`,此处迁移为策略类以纳入对比框架。
"""
from __future__ import annotations

from typing import List

from .base import Splitter


class FixedSizeSplitter(Splitter):
    """按固定字符窗口滑动切分,窗口间保留 overlap 重叠。"""

    name = "fixed"

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 需满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        chunks: List[str] = []
        start = 0
        step = max(1, self.chunk_size - self.overlap)
        while start < len(text):
            chunks.append(text[start:start + self.chunk_size])
            start += step
        return self._clean(chunks)
