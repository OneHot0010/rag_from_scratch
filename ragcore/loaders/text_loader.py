"""纯文本(.txt/.text)加载器。等价于阶段1 的 load_text,整篇作为一个 Document。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Document
from .base import DataLoader


class TextLoader(DataLoader):
    extensions = (".txt", ".text")

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> List[Document]:
        p = self._ensure_exists(path)
        text = p.read_text(encoding=self.encoding)
        doc = Document(content=text, metadata={"source": str(p), "format": "text"})
        return [doc] if doc else []
