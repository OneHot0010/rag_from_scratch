"""Excel 加载器(.xlsx/.xlsm,基于 openpyxl)。

策略:遍历每个 sheet,首行作为表头,其余每行 -> 一个 Document,
"列名: 值" 拼接为文本;元数据保留来源、sheet 名、行号,支持溯源。
只读模式(read_only)以降低大文件内存占用。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Document
from .base import DataLoader

try:
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover
    load_workbook = None


class ExcelLoader(DataLoader):
    extensions = (".xlsx", ".xlsm")

    def load(self, path: str | Path) -> List[Document]:
        if load_workbook is None:
            raise ImportError("需要 openpyxl,请先 `pip install openpyxl`。")
        p = self._ensure_exists(path)
        wb = load_workbook(filename=str(p), read_only=True, data_only=True)
        docs: List[Document] = []
        try:
            for ws in wb.worksheets:
                docs.extend(self._load_sheet(ws, source=str(p)))
        finally:
            wb.close()
        return docs

    def _load_sheet(self, ws, source: str) -> List[Document]:
        rows = ws.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration:
            return []
        headers = [str(h).strip() if h is not None else f"col{i}"
                   for i, h in enumerate(header)]
        docs: List[Document] = []
        for row_idx, values in enumerate(rows, start=2):
            content = self._row_to_text(headers, values)
            doc = Document(
                content=content,
                metadata={"source": source, "format": "excel",
                          "sheet": ws.title, "row": row_idx},
            )
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _row_to_text(headers: list, values) -> str:
        parts = []
        for h, v in zip(headers, values or ()):
            if v is not None and str(v).strip():
                parts.append(f"{h}: {str(v).strip()}")
        return "；".join(parts)
