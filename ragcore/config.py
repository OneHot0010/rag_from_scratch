"""集中配置:统一从 .env / 环境变量读取,供各模块复用。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_bool(val: str, default: bool = True) -> bool:
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "y")


@dataclass
class Settings:
    api_key: str = os.getenv("VOLCAN_API_KEY", "")
    base_url: str = os.getenv("VOLCAN_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    embed_model: str = os.getenv("EMBED_MODEL", "doubao-embedding-vision-251215")
    chat_model: str = os.getenv("CHAT_MODEL", "doubao-seed-2-1-pro-260628")
    doc_path: str = os.getenv("DOC_PATH", "sample.txt")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    overlap: int = int(os.getenv("OVERLAP", "50"))
    split_strategy: str = os.getenv("SPLIT_STRATEGY", "recursive")
    top_k: int = int(os.getenv("TOP_K", "3"))

    # ---- 阶段4:Embedding 模型选型 / 相似度度量 / 缓存 ----
    #: embedder 后端名。"ark" 表示用 settings.embed_model 的火山方舟模型;
    #: 也可直接写 "ark:<model>" 指定具体模型,或 "hash-offline" 走离线确定性向量。
    embed_backend: str = os.getenv("EMBED_BACKEND", "ark")
    #: 检索相似度度量:cosine / dot / l2(见 embedders.similarity)。
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
    #: 是否为 embedder 叠加缓存装饰器(内存 + 磁盘)。
    embed_cache: bool = _as_bool(os.getenv("EMBED_CACHE", ""), True)
    #: 磁盘缓存目录;空串表示仅用内存缓存(不持久化)。
    embed_cache_dir: str = os.getenv("EMBED_CACHE_DIR", ".cache/embeddings")


settings = Settings()
