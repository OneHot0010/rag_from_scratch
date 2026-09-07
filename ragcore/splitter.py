"""切分模块:固定长度切分(带 overlap 防止上下文断裂)。

对 Document 列表切分:保留每个 chunk 的来源元数据,支持溯源。
"""
from __future__ import annotations

from typing import List

from .models import Document


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return [c.strip() for c in chunks if c.strip()]


def split_documents(docs: List[Document], chunk_size: int = 300,
                    overlap: int = 50) -> List[Document]:
    out: List[Document] = []
    for doc in docs:
        for i, piece in enumerate(split_text(doc.content, chunk_size, overlap)):
            meta = dict(doc.metadata)
            meta["chunk"] = i
            out.append(Document(content=piece, metadata=meta))
    return out
