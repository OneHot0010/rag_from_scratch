"""递归字符切分(Recursive Character)。

思路:给定一组按「语义强度」排序的分隔符(段落 → 换行 → 句子 → 词 → 字符),
优先用最强的分隔符切;若切出的片段仍超过 chunk_size,就对该片段递归地
换下一级更细的分隔符继续切,直到不超限或分隔符用尽。

相比固定长度,它尽量保住语义边界(不轻易从句子/词中间断开),
是通用性最强的「结构无关」近似策略,也是工业界最常用的默认切分器。

合并阶段:递归切出的最小单元通常偏碎,再按 chunk_size 贪心合并相邻单元,
并在合并时保留 overlap,兼顾语义完整与检索粒度。
"""
from __future__ import annotations

from typing import List

from .base import Splitter

# 中英文通用的分隔符优先级:从「语义最完整」到「最细」。
_DEFAULT_SEPARATORS: List[str] = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ". ", "! ", "? ", "; ",
    "，", ", ",
    " ",
    "",
]


class RecursiveCharacterSplitter(Splitter):
    """按分隔符优先级递归切分,再贪心合并到目标长度。"""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        separators: List[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 需满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        splits = self._recursive_split(text, self.separators)
        merged = self._merge(splits)
        return self._clean(merged)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归产出「不超过 chunk_size 的最小语义单元」。"""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        sep = separators[0] if separators else ""
        rest = separators[1:] if len(separators) > 1 else []

        if sep == "":
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        pieces = self._split_keep(text, sep)
        out: List[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._recursive_split(piece, rest))
        return out

    @staticmethod
    def _split_keep(text: str, sep: str) -> List[str]:
        """按分隔符切分,并把分隔符保留在前一片段尾部(不丢标点/换行)。"""
        parts = text.split(sep)
        if sep == "":
            return parts
        out: List[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                out.append(part + sep)
            elif part:
                out.append(part)
        return out

    def _merge(self, splits: List[str]) -> List[str]:
        """把碎片贪心合并到接近 chunk_size,合并时保留 overlap。"""
        chunks: List[str] = []
        current = ""
        for piece in splits:
            if not current:
                current = piece
            elif len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                chunks.append(current)
                tail = current[-self.overlap:] if self.overlap else ""
                current = tail + piece
        if current:
            chunks.append(current)
        return chunks
