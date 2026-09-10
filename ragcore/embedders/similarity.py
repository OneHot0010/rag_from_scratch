"""相似度 / 距离度量:cosine / dot / L2。

阶段4 知识点:向量维度确定后,"两个向量有多像"取决于度量方式。三种最常用:

    - cosine(余弦相似度):只看方向不看长度,对文本长度/模长不敏感,
      是文本检索最常用度量。范围 [-1, 1],越大越相似。
    - dot(点积):既看方向也看模长。当向量已归一化时等价于 cosine;
      未归一化时,模长大的向量更易"胜出"。
    - l2(欧氏距离):越小越相似。为与前两者统一为"越大越相似",
      这里返回其负值(-distance),便于排序时统一取 Top-K 最大。

统一约定:所有度量都实现为 `score(query_vec, matrix) -> np.ndarray`,
返回 query 对矩阵每一行的**相似度分数(越大越相似)**,方便向量库直接 argsort。

与 splitters/loaders 一致地用注册表分派:新增度量=加一个函数+登记,不改上层。
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

_EPS = 1e-8


def _as_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    return m


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """余弦相似度:q·m / (|q|·|m|)。越大越相似,范围约 [-1, 1]。"""
    q = np.asarray(query_vec, dtype=np.float32)
    m = _as_matrix(matrix)
    denom = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) + _EPS) + _EPS
    return (m @ q) / denom


def dot_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """点积:q·m。越大越相似;对向量模长敏感。"""
    q = np.asarray(query_vec, dtype=np.float32)
    m = _as_matrix(matrix)
    return m @ q


def l2_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """欧氏(L2)距离的负值:-|q - m|。距离越小 -> 分数越大 -> 越相似。"""
    q = np.asarray(query_vec, dtype=np.float32)
    m = _as_matrix(matrix)
    dist = np.linalg.norm(m - q, axis=1)
    return -dist


#: 度量名 -> 打分函数。新增度量只需在此登记(开闭原则)。
_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cosine": cosine_scores,
    "dot": dot_scores,
    "l2": l2_scores,
}


def available_metrics() -> List[str]:
    """列出可用相似度度量名。"""
    return list(_METRICS)


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """按名字取相似度打分函数(统一语义:返回值越大越相似)。

    Args:
        name: cosine / dot / l2。

    Raises:
        ValueError: 未知度量名。
    """
    key = (name or "").lower()
    if key not in _METRICS:
        raise ValueError(
            f"未知相似度度量: {name!r}。可用: {', '.join(available_metrics())}"
        )
    return _METRICS[key]


def score(name: str, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """便捷入口:按度量名直接对矩阵打分。"""
    return get_metric(name)(query_vec, matrix)
