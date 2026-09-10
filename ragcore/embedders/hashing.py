"""离线确定性 Embedder(HashEmbedder)。

用途:在**没有 API Key** 的环境下也能跑通端到端流程、做单元测试与教学演示。
它不理解语义,但满足 embedder 的全部契约(稳定、可复现、可缓存),
因此非常适合验证"多模型可插拔 + 缓存 + 相似度度量"这套骨架本身是否正确。

原理(哈希词袋 + 随机投影思想的简化版):
    对文本做字符/词级 n-gram 哈希,散列到固定维度的桶里累加,再 L2 归一化。
    相同文本 -> 相同向量;有共同 n-gram 的文本 -> 向量更接近。
    这是一种确定性的稀疏-稠密映射,足以让"包含相同词的问句与 chunk"相互靠近,
    从而在无网络时演示检索链路。

⚠️ 仅用于离线演示/测试,召回质量远不及真实 Embedding 模型。
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from .base import BaseEmbedder

_TOKEN = re.compile(r"[0-9a-zA-Z一-鿿]+")


def _tokens(text: str) -> List[str]:
    """粗粒度分词:英文/数字按词,中文按单字(近似),再补 2-gram。"""
    words = _TOKEN.findall((text or "").lower())
    grams: List[str] = []
    for w in words:
        if _is_cjk(w):
            chars = list(w)
            grams.extend(chars)
            grams.extend(a + b for a, b in zip(chars, chars[1:]))  # 中文 2-gram
        else:
            grams.append(w)
    return grams or ["\x00"]


def _is_cjk(w: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in w)


class HashEmbedder(BaseEmbedder):
    """确定性哈希向量化,离线可用。

    Args:
        dim: 输出维度(默认 256),越大冲突越少。
    """

    name = "hash-offline"

    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0  # 有符号哈希,降低系统性偏置
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec
