"""向量化模块:调用 Ark multimodal embeddings 把文本转成向量。"""
from __future__ import annotations

from typing import List

import numpy as np
from volcenginesdkarkruntime import Ark

from .config import settings


class Embedder:
    def __init__(self, client: Ark | None = None, model: str | None = None) -> None:
        self.client = client or Ark(api_key=settings.api_key, base_url=settings.base_url)
        self.model = model or settings.embed_model

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        res: List[np.ndarray] = []
        for text in texts:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}],
            )
            res.append(np.array(resp.data.embedding, dtype=np.float32))
        return res

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
