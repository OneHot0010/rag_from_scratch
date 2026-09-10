"""Embedding 模型的抽象接口。

阶段4 目标:把「向量化(Embedding)」从写死的单一模型,抽象为可插拔、
可横向对比的模型族——正如阶段2 对 loaders、阶段3 对 splitters 所做的那样。

统一契约:
    每个 embedder 声明自己的名字(name)与向量维度(dim),并实现
    `embed(texts) -> List[np.ndarray]`。基类据此提供:
        - embed_one(text):单条便捷封装;
        - embed_documents(docs):对 Document 批量向量化(顺序与输入对齐)。

设计要点(对标 splitters/loaders):
    - 开闭原则:新增模型 = 新增一个子类 + 在 registry 登记,不改核心流程。
    - 稠密向量:所有实现都产出稠密浮点向量(np.float32),供相似度度量比较。
    - 归一化解耦:是否 L2 归一化交由相似度度量(similarity.py)处理,
      embedder 只负责"把文本映射成向量",职责单一。
"""
from __future__ import annotations

import abc
from typing import ClassVar, List

import numpy as np

from ..models import Document


class BaseEmbedder(abc.ABC):
    """所有 Embedding 模型的统一接口。

    Attributes:
        name: 模型标识(唯一),供 registry 分派、实验报告与缓存 key 使用。
        dim:  输出向量维度。0 表示"运行前未知/动态"(如远程模型首调后才确定)。
    """

    #: 模型名(唯一标识)。
    name: ClassVar[str] = ""
    #: 向量维度;0 表示未知(远程模型可在首次调用后回填)。
    dim: ClassVar[int] = 0

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """把一批文本向量化为稠密向量列表。

        Args:
            texts: 待向量化文本列表。

        Returns:
            与输入等长、顺序对齐的向量列表(每个为 np.ndarray[float32])。
        """
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """单条文本向量化的便捷封装。"""
        return self.embed([text])[0]

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """对 Document 列表批量向量化(取其 content,顺序对齐)。"""
        return self.embed([d.content for d in docs])

    def __repr__(self) -> str:  # pragma: no cover - 仅调试展示
        return f"<{self.__class__.__name__} name={self.name!r} dim={self.dim}>"
