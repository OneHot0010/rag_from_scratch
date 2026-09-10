"""Embedding 缓存装饰器(CachedEmbedder)。

阶段4 核心交付之一:批量向量化 + 缓存机制。

为什么需要缓存:
    - Embedding 调用有**耗时与成本**;同一份 chunk 在反复实验/多次建库时会被
      重复向量化,浪费 API 额度与时间。
    - 缓存 key = (模型名, 文本内容) 的哈希。只要模型与文本不变,向量即可复用;
      换模型会自然命中不同 key,互不污染。

设计(装饰器模式,对上层透明):
    CachedEmbedder 包裹任意 BaseEmbedder,实现同样的接口。
    - 内存层(dict):进程内即时命中;
    - 磁盘层(可选,npz/json):跨进程/跨次运行持久化,重启不丢。
    - 批量去重:一次 embed 里重复文本只算一次,未命中的才真正下发底层模型,
      再按原顺序回填——既省调用又保持返回顺序对齐。

统计:命中/未命中计数(hits/misses)供实验脚本量化"缓存收益"。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import BaseEmbedder


def _key(model: str, text: str) -> str:
    raw = f"{model}\x00{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class CachedEmbedder(BaseEmbedder):
    """为任意 embedder 叠加内存 + 磁盘缓存与批量去重。

    Args:
        inner:     被包裹的底层 embedder(真正产生向量的那个)。
        cache_dir: 磁盘缓存目录;None 表示仅用内存缓存(不持久化)。
    """

    def __init__(self, inner: BaseEmbedder, cache_dir: str | Path | None = ".cache/embeddings") -> None:
        self.inner = inner
        self._mem: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._dir: Path | None = None
        if cache_dir is not None:
            self._dir = Path(cache_dir) / _safe(inner.name)
            self._dir.mkdir(parents=True, exist_ok=True)

    # 保持与底层一致的身份,便于报告与再包裹。
    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    # ---- 磁盘读写(单条一个 .npy,key 作文件名;简单可靠、便于增量)----
    def _disk_path(self, key: str) -> Path | None:
        return None if self._dir is None else self._dir / f"{key}.npy"

    def _load_disk(self, key: str) -> np.ndarray | None:
        p = self._disk_path(key)
        if p is not None and p.exists():
            try:
                return np.load(p)
            except Exception:  # noqa: BLE001 - 缓存损坏则视为未命中
                return None
        return None

    def _save_disk(self, key: str, vec: np.ndarray) -> None:
        p = self._disk_path(key)
        if p is not None:
            try:
                np.save(p, vec)
            except Exception:  # noqa: BLE001 - 落盘失败不应影响主流程
                pass

    def _get_cached(self, key: str) -> np.ndarray | None:
        if key in self._mem:
            return self._mem[key]
        vec = self._load_disk(key)
        if vec is not None:
            self._mem[key] = vec
        return vec

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        model = self.inner.name
        keys = [_key(model, t) for t in texts]

        # 1) 先查缓存,记录未命中的"唯一文本"(去重)。
        results: List[np.ndarray | None] = [None] * len(texts)
        to_compute: Dict[str, int] = {}  # key -> 该 key 对应的某个 texts 下标
        for i, (t, k) in enumerate(zip(texts, keys)):
            cached = self._get_cached(k)
            if cached is not None:
                results[i] = cached
                self.hits += 1
            else:
                self.misses += 1
                if k not in to_compute:
                    to_compute[k] = i  # 仅登记一次,天然去重

        # 2) 未命中的唯一文本批量下发底层模型。
        if to_compute:
            uniq_keys = list(to_compute)
            uniq_texts = [texts[to_compute[k]] for k in uniq_keys]
            vectors = self.inner.embed(uniq_texts)
            for k, vec in zip(uniq_keys, vectors):
                self._mem[k] = vec
                self._save_disk(k, vec)

        # 3) 回填所有位置(命中缓存 + 新算的),保持与输入顺序对齐。
        for i, k in enumerate(keys):
            if results[i] is None:
                results[i] = self._mem[k]
        return [r for r in results]  # type: ignore[list-item]

    def stats(self) -> Dict[str, int]:
        """返回缓存命中统计,供实验脚本量化收益。"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_pct": round(100 * self.hits / total, 1) if total else 0.0,
        }

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


def _safe(name: str) -> str:
    """把模型名转成安全的目录名。"""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "default"
