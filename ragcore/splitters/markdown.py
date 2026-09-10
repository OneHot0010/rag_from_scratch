"""Markdown 结构感知切分(Structure-aware)。

思路:顺着文档天然结构走——按 ATX 标题(``#``~``######``)划分小节,
每个 chunk 携带其所在的**标题层级路径**(如 ``安装 > 依赖``)作为上下文锚点,
写入 metadata,缓解「脱离标题的裸段落语义不完整」的问题。

超长小节回退到递归字符切分,避免单节撑爆 chunk;代码块(fenced)
内的 `#` 不被误判为标题。
"""
from __future__ import annotations

import re
from typing import List

from .base import Splitter
from .recursive import RecursiveCharacterSplitter
from ..models import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkdownSplitter(Splitter):
    """按标题层级切分 Markdown,保留标题路径元数据。"""

    name = "markdown"

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = RecursiveCharacterSplitter(chunk_size, overlap)

    def split_text(self, text: str) -> List[str]:
        """仅返回正文 chunk(不含元数据)。溯源路径见 split_documents。"""
        return [c for c, _ in self._split_sections(text or "")]

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """重写:为每个 chunk 注入 heading_path 元数据。"""
        out: List[Document] = []
        for doc in docs:
            sections = self._split_sections(doc.content)
            idx = 0
            for content, heading_path in sections:
                for piece in self._fit(content):
                    meta = dict(doc.metadata)
                    meta["chunk"] = idx
                    meta["splitter"] = self.name
                    if heading_path:
                        meta["heading_path"] = heading_path
                    chunk_doc = Document(content=piece, metadata=meta)
                    if chunk_doc:
                        out.append(chunk_doc)
                        idx += 1
        return out

    def _fit(self, content: str) -> List[str]:
        """小节过长则回退递归切分,否则整节作为一个 chunk。"""
        if len(content) <= self.chunk_size:
            return self._clean([content])
        return self._fallback.split_text(content)

    def _split_sections(self, text: str) -> List[tuple]:
        """把 Markdown 切成 (小节正文含标题, 标题路径) 列表。"""
        lines = text.split("\n")
        sections: List[tuple] = []
        stack: List[tuple] = []
        buf: List[str] = []
        in_fence = False

        def flush() -> None:
            body = "\n".join(buf).strip()
            if body:
                path = " > ".join(t for _, t in stack)
                sections.append((body, path))

        for line in lines:
            if _FENCE.match(line):
                in_fence = not in_fence
                buf.append(line)
                continue
            m = _HEADING.match(line) if not in_fence else None
            if m:
                flush()
                buf = [line]
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
            else:
                buf.append(line)
        flush()
        return sections
