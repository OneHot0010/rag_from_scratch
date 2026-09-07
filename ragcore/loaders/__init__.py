"""数据接入层:每种格式一个 DataLoader 接口实现。"""
from .base import DataLoader
from .text_loader import TextLoader
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .stub_loaders import PDFLoader, WordLoader, MarkdownLoader, HTMLLoader
from .registry import LoaderRegistry, build_default_registry

__all__ = [
    "DataLoader",
    "TextLoader",
    "CSVLoader",
    "ExcelLoader",
    "PDFLoader",
    "WordLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "LoaderRegistry",
    "build_default_registry",
]
