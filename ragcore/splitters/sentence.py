"""按句 / 按段切分(Sentence / Paragraph)。

- 段落切分:按空行(``\n\n``)划分自然段,再按 chunk_size 贪心合并相邻段。
- 句子切分:用中英文标点正则分句,再贪心合并相邻句到目标长度,句间带 overlap。

不依赖 NLTK/spaCy 等重库(阶段3 保持 from-scratch、可离线运行),
用正则近似分句;对多数中英文文档已足够,极端缩写/小数点场景会有个别误切。
"""
from __future__ import annotations

import re
from typing import List

from .base import Splitter

# 在中英文句末标点后切分,并把标点留在句尾(前瞻)。
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")


class SentenceSplitter(Splitter):
    """按句分割,再贪心合并相邻句到接近 chunk_size。"""

    name = "sentence"

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 需满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        sentences = [s for s in _SENTENCE_END.split(text or "") if s and s.strip()]
        return self._clean(_greedy_merge(sentences, self.chunk_size, self.overlap))


class ParagraphSplitter(Splitter):
    """按空行分段,超长段落回退到句子切分,再贪心合并到接近 chunk_size。"""

    name = "paragraph"

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 需满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._sentence = SentenceSplitter(chunk_size, overlap)

    def split_text(self, text: str) -> List[str]:
        paragraphs = [p for p in re.split(r"\n\s*\n", text or "") if p and p.strip()]
        units: List[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(para.strip())
            else:
                units.extend(self._sentence.split_text(para))
        return self._clean(_greedy_merge(units, self.chunk_size, self.overlap))


def _greedy_merge(units: List[str], chunk_size: int, overlap: int) -> List[str]:
    """把语义单元(句/段)贪心拼到接近 chunk_size,拼接处保留 overlap。"""
    chunks: List[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = current + " " + unit if current else unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + unit).strip() if tail else unit
    if current:
        chunks.append(current)
    return chunks
