"""CSV 加载器(标准库 csv,无第三方依赖)。

策略:每一行 -> 一个 Document,把 "列名: 值" 拼成自然语言式文本,
便于 embedding 理解语义;元数据保留来源文件与行号,支持溯源。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from ..models import Document
from .base import DataLoader


class CSVLoader(DataLoader):
    extensions = (".csv",)

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> List[Document]:
        p = self._ensure_exists(path)
        docs: List[Document] = []
        with p.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row_idx, row in enumerate(reader, start=2):  # 第1行为表头
                content = self._row_to_text(row, headers)
                doc = Document(
                    content=content,
                    metadata={"source": str(p), "format": "csv", "row": row_idx},
                )
                if doc:
                    docs.append(doc)
        return docs

    @staticmethod
    def _row_to_text(row: dict, headers: list) -> str:
        parts = []
        for h in headers:
            val = (row.get(h) or "").strip()
            if val:
                parts.append(f"{h}: {val}")
        return "；".join(parts)
