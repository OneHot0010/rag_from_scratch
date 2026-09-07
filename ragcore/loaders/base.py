"""数据接入层的抽象接口。

阶段2 目标:每一种数据格式都抽象为一个 DataLoader 接口。
本文件定义统一契约,具体格式由子类实现:
    - 已实现: TextLoader / CSVLoader / ExcelLoader
    - 仅留接口(抛 NotImplementedError): PDF / Word / Markdown / HTML

约定:
    每个 loader 声明自己支持的扩展名(extensions),并实现 load()
    把单个文件解析为 List[Document]。上层通过 registry 按扩展名分派,
    因此新增格式只需实现一个子类并注册,无需改动核心流程(开闭原则)。
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import ClassVar, List, Tuple

from ..models import Document


class DataLoader(abc.ABC):
    """所有数据格式加载器的统一接口。"""

    #: 该 loader 支持的文件扩展名(小写,含点),供 registry 分派使用。
    extensions: ClassVar[Tuple[str, ...]] = ()

    @abc.abstractmethod
    def load(self, path: str | Path) -> List[Document]:
        """把单个文件解析为 Document 列表。

        Args:
            path: 文件路径。

        Returns:
            解析出的 Document 列表(可能为多个,如 Excel 的多行/多 sheet)。

        Raises:
            FileNotFoundError: 文件不存在。
            NotImplementedError: 该格式的 loader 尚未实现。
        """
        raise NotImplementedError

    def supports(self, path: str | Path) -> bool:
        """判断该 loader 是否支持给定文件(按扩展名)。"""
        return Path(path).suffix.lower() in self.extensions

    @staticmethod
    def _ensure_exists(path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        return p
