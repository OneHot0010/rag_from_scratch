"""尚未实现的格式:仅声明接口,占位待后续阶段实现。

阶段2 只落地 Excel/CSV(及 text)。PDF/Word/Markdown/HTML 先保留
统一接口与扩展名声明,load() 抛 NotImplementedError,清晰暴露
"接口已定义、实现待补"的边界,方便渐进式扩展而不破坏架构。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Document
from .base import DataLoader


class _NotImplementedLoader(DataLoader):
    """占位基类:统一抛出未实现错误。"""

    format_name: str = "unknown"

    def load(self, path: str | Path) -> List[Document]:
        raise NotImplementedError(
            f"{self.format_name} 加载器尚未实现(接口已预留)。"
            f"当前阶段仅支持 Excel/CSV/Text。"
        )


class PDFLoader(_NotImplementedLoader):
    extensions = (".pdf",)
    format_name = "PDF"


class WordLoader(_NotImplementedLoader):
    extensions = (".docx", ".doc")
    format_name = "Word"


class MarkdownLoader(_NotImplementedLoader):
    extensions = (".md", ".markdown")
    format_name = "Markdown"


class HTMLLoader(_NotImplementedLoader):
    extensions = (".html", ".htm")
    format_name = "HTML"
