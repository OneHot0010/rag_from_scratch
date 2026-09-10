"""RAG 编排:把加载->切分->向量化->建库->检索->生成串起来。

对外只暴露 RAGPipeline,屏蔽各模块细节。数据接入通过 LoaderRegistry
按扩展名自动分派;切分通过 splitters 注册表按策略名分派(阶段3);
向量化(阶段4)通过 embedders 注册表按后端名分派,可插拔多模型 + 缓存,
相似度度量(cosine/dot/l2)亦可配置。默认值见 settings。
"""
from __future__ import annotations

from typing import List, Tuple

from volcenginesdkarkruntime import Ark

from .config import settings
from .models import Document
from .loaders import build_default_registry, LoaderRegistry
from .splitters import build_splitter, EMBEDDING_STRATEGIES
from .embedders import build_embedder
from .embedders.base import BaseEmbedder
from .vector_store import InMemoryVectorStore
from .generator import Generator


class RAGPipeline:
    def __init__(
        self,
        registry: LoaderRegistry | None = None,
        split_strategy: str | None = None,
        embed_backend: str | None = None,
        similarity_metric: str | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        # Generator 仍复用一个 Ark client;embedder 若走 ark 后端也复用它。
        client = Ark(api_key=settings.api_key, base_url=settings.base_url)
        self.registry = registry or build_default_registry()
        self.generator = Generator(client=client)

        # 向量化后端(阶段4):显式传入 embedder > 后端名 > 配置默认。
        if embedder is not None:
            self.embedder = embedder
        else:
            backend = embed_backend or settings.embed_backend
            cache_dir = settings.embed_cache_dir or None
            self.embedder = build_embedder(
                backend,
                cache=settings.embed_cache,
                cache_dir=cache_dir,
                client=client,
            )

        # 相似度度量(阶段4):显式传入 > 配置默认。
        self.metric = (similarity_metric or settings.similarity_metric).lower()
        self.store = InMemoryVectorStore(metric=self.metric)

        # 切分策略:显式传入 > 配置默认。semantic 需注入 embedder。
        self.strategy = (split_strategy or settings.split_strategy).lower()
        splitter_embedder = self.embedder if self.strategy in EMBEDDING_STRATEGIES else None
        self.splitter = build_splitter(
            self.strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.overlap,
            embedder=splitter_embedder,
        )

    def build_index(self, paths: List[str]) -> int:
        docs = self.registry.load_paths(paths)
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            raise ValueError("未从输入文件解析出任何内容。")
        vectors = self.embedder.embed([c.content for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> List[Tuple[Document, float]]:
        q = self.embedder.embed_one(query)
        return self.store.search(q, k or settings.top_k)

    def ask(self, query: str, k: int | None = None) -> Tuple[str, List[Tuple[Document, float]]]:
        hits = self.retrieve(query, k)
        answer = self.generator.generate(query, hits)
        return answer, hits
