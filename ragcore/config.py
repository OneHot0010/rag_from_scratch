"""集中配置:统一从 .env / 环境变量读取,供各模块复用。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    api_key: str = os.getenv("VOLCAN_API_KEY", "")
    base_url: str = os.getenv("VOLCAN_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    embed_model: str = os.getenv("EMBED_MODEL", "doubao-embedding-vision-251215")
    chat_model: str = os.getenv("CHAT_MODEL", "doubao-seed-2-1-pro-260628")
    doc_path: str = os.getenv("DOC_PATH", "sample.txt")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    overlap: int = int(os.getenv("OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "3"))


settings = Settings()
