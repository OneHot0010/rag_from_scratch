"""语义切分(Semantic Chunking)。

思路:先按句分割,计算相邻句的 embedding 余弦相似度,在**相似度骤降的断点**
处切开——即「话题发生转移」的地方,使每个 chunk 主题内聚。相比结构无关的
字符切分,更贴合语义边界。

代价:需要对每个句子调用 Embedding,较慢且产生 API 成本,故设为可选策略
(需注入 Embedder)。断点阈值用「相似度分布的分位数」自适应,避免手调死阈值。
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from .base import Splitter

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")


class SemanticSplitter(Splitter):
    """基于句间 embedding 相似度断点的语义切分。

    Args:
        embedder: 任何提供 ``embed(List[str]) -> List[np.ndarray]`` 的对象
                  (复用 ragcore.embedder.Embedder)。
        breakpoint_percentile: 断点阈值分位数(0~100)。相邻句距离
                  (1 - cos) 超过该分位数时视为话题转移,在此切开。
        max_chunk_size: 单 chunk 字符上限,超过则强制断开,防止过长。
        min_sentences: 每个 chunk 至少包含的句子数,避免过碎。
    """

    name = "semantic"

    def __init__(
        self,
        embedder,
        breakpoint_percentile: float = 90.0,
        max_chunk_size: int = 800,
        min_sentences: int = 1,
    ) -> None:
        if embedder is None:
            raise ValueError("SemanticSplitter 需要注入 embedder")
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_size = max_chunk_size
        self.min_sentences = max(1, min_sentences)

    def split_text(self, text: str) -> List[str]:
        sentences = [s.strip() for s in _SENTENCE_END.split(text or "") if s and s.strip()]
        if len(sentences) <= 1:
            return self._clean(sentences)

        vecs = self.embedder.embed(sentences)
        distances = [
            1.0 - _cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)
        ]
        threshold = float(np.percentile(distances, self.breakpoint_percentile)) \
            if distances else 1.0

        chunks: List[str] = []
        current: List[str] = [sentences[0]]
        for i, dist in enumerate(distances):
            nxt = sentences[i + 1]
            over_size = len(" ".join(current)) + len(nxt) > self.max_chunk_size
            breakpoint = dist > threshold and len(current) >= self.min_sentences
            if breakpoint or over_size:
                chunks.append(" ".join(current))
                current = [nxt]
            else:
                current.append(nxt)
        if current:
            chunks.append(" ".join(current))
        return self._clean(chunks)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
