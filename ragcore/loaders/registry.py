"""Loader 注册表:按文件扩展名分派到对应 DataLoader。

新增格式的唯一改动点:在 _DEFAULT_LOADERS 里注册一个实例。
上层(pipeline)只调用 load_file / load_paths,无需感知具体格式(开闭原则)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..models import Document
from .base import DataLoader
from .text_loader import TextLoader
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .stub_loaders import PDFLoader, WordLoader, MarkdownLoader, HTMLLoader


class LoaderRegistry:
    def __init__(self) -> None:
        self._by_ext: Dict[str, DataLoader] = {}

    def register(self, loader: DataLoader) -> None:
        for ext in loader.extensions:
            self._by_ext[ext.lower()] = loader

    def get(self, path: str | Path) -> DataLoader:
        ext = Path(path).suffix.lower()
        if ext not in self._by_ext:
            raise ValueError(
                f"不支持的文件格式: {ext or '(无扩展名)'}。"
                f"已支持: {', '.join(sorted(self._by_ext))}"
            )
        return self._by_ext[ext]

    def supported_extensions(self) -> List[str]:
        return sorted(self._by_ext)

    def load_file(self, path: str | Path) -> List[Document]:
        return self.get(path).load(path)

    def load_paths(self, paths: List[str | Path]) -> List[Document]:
        docs: List[Document] = []
        for p in paths:
            docs.extend(self.load_file(p))
        return docs


def build_default_registry() -> LoaderRegistry:
    """构建默认注册表:已实现 + 预留接口全部登记。"""
    reg = LoaderRegistry()
    for loader in (
        TextLoader(),
        CSVLoader(),
        ExcelLoader(),
        # 以下为预留接口,调用 load() 会抛 NotImplementedError
        PDFLoader(),
        WordLoader(),
        MarkdownLoader(),
        HTMLLoader(),
    ):
        reg.register(loader)
    return reg
