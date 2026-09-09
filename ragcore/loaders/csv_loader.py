"""CSV 加载器(标准库 csv,无第三方依赖)。

阶段2.1 增强(P0 正确性):
  - 编码嗅探:优先按 BOM 判定(utf-8-sig/utf-16),否则在候选编码
    (utf-8, gbk, gb18030, latin-1)中择可解码者,避免中文乱码;
  - 方言嗅探:用 csv.Sniffer 猜分隔符/引号(兼容 , ; \t | 等),
    失败则回退默认逗号;
  - 可配置表头行:skip_rows 跳过前置说明行,再取表头;
  - 类型规范化 + 跳过空行(复用 _tabular)。

每行 -> 一个 Document,"列名: 值" 拼成自然语言式文本;
元数据保留来源、行号、检测到的编码/分隔符,便于排查。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import Document
from .base import DataLoader
from ._tabular import clean_headers, row_to_text, is_blank_row

# 中文环境常见编码候选(顺序即优先级)
_CANDIDATE_ENCODINGS = ("utf-8", "gbk", "gb18030", "latin-1")
_BOM_ENCODINGS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


class CSVLoader(DataLoader):
    extensions = (".csv",)

    def __init__(self, encoding: Optional[str] = None, delimiter: Optional[str] = None,
                 skip_rows: int = 0) -> None:
        """
        Args:
            encoding:  指定编码;None 则自动嗅探。
            delimiter: 指定分隔符;None 则自动嗅探。
            skip_rows: 表头之前需要跳过的前置行数(标题/说明)。
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.skip_rows = max(0, skip_rows)

    def load(self, path: str | Path) -> List[Document]:
        p = self._ensure_exists(path)
        raw = p.read_bytes()
        encoding = self.encoding or self._detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        delimiter = self.delimiter or self._detect_delimiter(text)

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        all_rows = list(reader)
        # 跳过前置说明行
        rows = all_rows[self.skip_rows:]
        if not rows:
            return []

        headers = clean_headers(rows[0])
        docs: List[Document] = []
        for offset, values in enumerate(rows[1:]):
            row_idx = self.skip_rows + offset + 2  # 1-based 真实行号
            if is_blank_row(values):
                continue
            content = row_to_text(headers, values)
            doc = Document(
                content=content,
                metadata={"source": str(p), "format": "csv", "row": row_idx,
                          "encoding": encoding, "delimiter": delimiter},
            )
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _detect_encoding(raw: bytes) -> str:
        for bom, enc in _BOM_ENCODINGS:
            if raw.startswith(bom):
                return enc
        for enc in _CANDIDATE_ENCODINGS:
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        return "utf-8"  # 兜底(decode 时 errors=replace)

    @staticmethod
    def _detect_delimiter(text: str) -> str:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","
