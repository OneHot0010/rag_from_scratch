"""RAG 编排:把加载->切分->向量化->建库->检索->生成串起来。

对外只暴露 RAGPipeline,屏蔽各模块细节。数据接入通过 LoaderRegistry
按扩展名自动分派,因此 build_index 可接受任意已支持格式的文件。
"""
from __future__ import annotations

from typing import List, Tuple

from volcenginesdkarkruntime import Ark

from .config import settings
from .models import Document
from .loaders import build_default_registry, LoaderRegistry
from .splitter import split_documents
from .embedder import Embedder
from .vector_store import InMemoryVectorStore
from .generator import Generator


class RAGPipeline:
    def __init__(self, registry: LoaderRegistry | None = None) -> None:
        # Embedder 与 Generator 复用同一个 Ark client。
        client = Ark(api_key=settings.api_key, base_url=settings.base_url)
        self.registry = registry or build_default_registry()
        self.embedder = Embedder(client=client)
        self.generator = Generator(client=client)
        self.store = InMemoryVectorStore()

    def build_index(self, paths: List[str]) -> int:
        docs = self.registry.load_paths(paths)
        chunks = split_documents(docs, settings.chunk_size, settings.overlap)
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
