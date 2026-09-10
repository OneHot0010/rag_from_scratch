"""Embedding 模型层:阶段4 · 把「向量化」抽象为可插拔、可对比的模型族。

已实现:
    - hash-offline   离线确定性哈希向量(无需 API Key,测试/演示用)
    - ark:<model>    火山方舟下任意 Embedding 模型(切 model 名即切模型)

配套能力:
    - CachedEmbedder   缓存装饰器(内存 + 磁盘,批量去重,命中统计)
    - similarity       cosine / dot / l2 三种相似度度量及注册表
    - registry         按名字构建模型(开闭原则),可选叠加缓存
"""
from .base import BaseEmbedder
from .hashing import HashEmbedder
from .ark import ArkEmbedder
from .cache import CachedEmbedder
from .registry import (
    build_embedder,
    available_embedders,
    OFFLINE_EMBEDDERS,
)
from .similarity import (
    available_metrics,
    get_metric,
    score,
    cosine_scores,
    dot_scores,
    l2_scores,
)

__all__ = [
    "BaseEmbedder",
    "HashEmbedder",
    "ArkEmbedder",
    "CachedEmbedder",
    "build_embedder",
    "available_embedders",
    "OFFLINE_EMBEDDERS",
    "available_metrics",
    "get_metric",
    "score",
    "cosine_scores",
    "dot_scores",
    "l2_scores",
]
