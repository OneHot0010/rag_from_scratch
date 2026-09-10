"""火山方舟(Ark)Embedding 适配器。

把阶段1/2 里写死的 Ark multimodal embeddings 调用,收敛为符合
`BaseEmbedder` 契约的一个实现,从而能与其它模型在实验里横向对比、
并叠加缓存装饰器(见 cache.py)。

说明:
    - 通过 model 名区分不同商用/开源模型(同一 Ark 接口下切换 model 即可),
      因此一个类可代表"多个模型",在对比实验里用不同 model 实例化即可。
    - dim 在首次成功调用后回填(远程模型维度运行时才确定)。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..config import settings
from .base import BaseEmbedder


class ArkEmbedder(BaseEmbedder):
    """基于火山方舟 multimodal_embeddings 的稠密向量化实现。

    Args:
        model:   Embedding 模型名(默认取 settings.embed_model)。
        client:  可注入已建好的 Ark client(便于复用连接与测试),
                 不传则惰性创建。
    """

    def __init__(self, model: str | None = None, client=None) -> None:
        self._model = model or settings.embed_model
        self._client = client
        self.dim = 0  # 首次调用后回填

    @property
    def name(self) -> str:  # 用具体模型名作为标识,便于缓存 key 与报告区分
        return self._model

    def _ensure_client(self):
        if self._client is None:
            from volcenginesdkarkruntime import Ark  # 惰性导入,离线场景不强依赖

            self._client = Ark(api_key=settings.api_key, base_url=settings.base_url)
        return self._client

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        client = self._ensure_client()
        out: List[np.ndarray] = []
        for text in texts:
            resp = client.multimodal_embeddings.create(
                model=self._model,
                input=[{"type": "text", "text": text}],
            )
            vec = np.array(resp.data.embedding, dtype=np.float32)
            if not self.dim:
                self.dim = int(vec.shape[0])
            out.append(vec)
        return out
