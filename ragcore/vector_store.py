"""内存向量库:保存 chunk 向量,支持 Top-K 余弦相似检索。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .models import Document


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._docs: List[Document] = []
        self._vecs: np.ndarray | None = None

    def add(self, docs: List[Document], vectors: List[np.ndarray]) -> None:
        self._docs.extend(docs)
        mat = np.array(vectors, dtype=np.float32)
        self._vecs = mat if self._vecs is None else np.vstack([self._vecs, mat])

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        if self._vecs is None or len(self._docs) == 0:
            return []
        sims = self._vecs @ query_vec / (
            np.linalg.norm(self._vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        top = sims.argsort()[::-1][:k]
        return [(self._docs[i], float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self._docs)
