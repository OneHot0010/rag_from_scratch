"""内存向量库:保存 chunk 向量,支持 Top-K 相似检索。

阶段4 起相似度度量可插拔:构造时传入 metric(cosine / dot / l2),
默认 cosine,与阶段1~3 行为一致(向后兼容)。统一约定"分值越大越相似",
因此 l2 返回的是负距离(见 embedders.similarity)。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .models import Document
from .embedders.similarity import get_metric


class InMemoryVectorStore:
    def __init__(self, metric: str = "cosine") -> None:
        self._docs: List[Document] = []
        self._vecs: np.ndarray | None = None
        self.metric = metric
        self._score = get_metric(metric)  # 校验并取度量函数

    def add(self, docs: List[Document], vectors: List[np.ndarray]) -> None:
        self._docs.extend(docs)
        mat = np.array(vectors, dtype=np.float32)
        self._vecs = mat if self._vecs is None else np.vstack([self._vecs, mat])

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        if self._vecs is None or len(self._docs) == 0:
            return []
        sims = self._score(np.asarray(query_vec, dtype=np.float32), self._vecs)
        top = sims.argsort()[::-1][:k]
        return [(self._docs[i], float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self._docs)
