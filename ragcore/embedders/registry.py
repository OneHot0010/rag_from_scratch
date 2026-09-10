"""Embedder 注册表:按名字构建 Embedding 模型,并可选叠加缓存。

与 loaders/splitters 的 registry 对称:新增模型的唯一改动点是在这里登记
一个工厂。上层(pipeline / 实验脚本)只按名字取模型,无需感知具体实现
(开闭原则)。

命名约定:
    - "hash-offline"            离线确定性模型,无需 API Key(测试/演示)。
    - "ark:<model_name>"        火山方舟下的任意 Embedding 模型,冒号后为具体
                                model 名;省略则用 settings.embed_model。
      例如 "ark:doubao-embedding-large" 与 "ark:doubao-embedding" 视为两个模型。
"""
from __future__ import annotations

from typing import List

from .base import BaseEmbedder
from .cache import CachedEmbedder
from .hashing import HashEmbedder
from .ark import ArkEmbedder

#: 不依赖 API、可离线构建的模型名(供实验脚本判断是否需要 Key)。
OFFLINE_EMBEDDERS = (HashEmbedder.name,)


def available_embedders(include_offline: bool = True) -> List[str]:
    """列出内置可直接命名的模型(不含 ark 动态 model)。"""
    names = ["ark:<model_name>"]
    if include_offline:
        names = list(OFFLINE_EMBEDDERS) + names
    return names


def build_embedder(
    name: str,
    *,
    cache: bool = False,
    cache_dir: str | None = ".cache/embeddings",
    client=None,
    dim: int = 256,
) -> BaseEmbedder:
    """按名字构建 embedder。

    Args:
        name:      模型名。见 available_embedders();支持 "ark:<model>"。
        cache:     是否叠加 CachedEmbedder 缓存装饰器。
        cache_dir: 磁盘缓存目录(cache=True 时生效;None 则仅内存)。
        client:    可注入的 Ark client(仅 ark:* 生效,便于复用/测试)。
        dim:       hash-offline 的维度。

    Raises:
        ValueError: 未知模型名。
    """
    key = (name or "").strip()
    low = key.lower()

    if low == HashEmbedder.name:
        emb: BaseEmbedder = HashEmbedder(dim=dim)
    elif low.startswith("ark:") or low == "ark":
        model = key.split(":", 1)[1].strip() if ":" in key else None
        emb = ArkEmbedder(model=model or None, client=client)
    else:
        raise ValueError(
            f"未知 Embedding 模型: {name!r}。"
            f"可用: {', '.join(available_embedders())}"
        )

    if cache:
        emb = CachedEmbedder(emb, cache_dir=cache_dir)
    return emb
