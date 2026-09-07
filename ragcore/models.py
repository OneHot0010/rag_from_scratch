"""数据模型:统一的文档表示。

所有 DataLoader 都把各种原始格式(txt/csv/excel/pdf...)解析为
`Document` 列表。Document 携带正文与元数据(来源、页码、行号、
sheet 名等),使检索结果可溯源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """RAG 中流转的最小知识单元。

    Attributes:
        content:  文本正文,后续会被切分/向量化。
        metadata: 溯源信息,例如 {"source": "a.xlsx", "sheet": "Sheet1",
                  "row": 12}。不同 loader 可自由扩展键。
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = (self.content or "").strip()

    def __bool__(self) -> bool:
        return bool(self.content)
