"""切分策略的抽象接口。

阶段3 目标:把「切分(Chunking)」从单一固定长度,抽象为可插拔的策略族,
便于按数据类型选型与横向对比。

统一契约:
    每个策略声明自己的名字(name),并实现 `split_text(text) -> List[str]`。
    基类提供 `split_documents(docs)`:对 Document 列表逐个切分,自动继承并
    补充 chunk 元数据(chunk 序号 + 策略名),保证检索结果可溯源。

新增策略只需实现一个子类并注册,无需改动核心流程(开闭原则),
与 loaders 的设计保持一致。
"""
from __future__ import annotations

import abc
from typing import ClassVar, List

from ..models import Document


class Splitter(abc.ABC):
    """所有切分策略的统一接口。"""

    #: 策略名(唯一标识),供 registry 分派、实验报告与元数据标注使用。
    name: ClassVar[str] = ""

    @abc.abstractmethod
    def split_text(self, text: str) -> List[str]:
        """把一段纯文本切成若干 chunk(字符串)。

        Args:
            text: 待切分的正文。

        Returns:
            chunk 字符串列表(已去除空白片段)。
        """
        raise NotImplementedError

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """对 Document 列表逐个切分,保留并扩展溯源元数据。

        每个产出 chunk 的 metadata 会在原 Document 基础上追加:
            - ``chunk``:该 chunk 在所属 Document 内的序号(从 0 起);
            - ``splitter``:产出它的切分策略名。

        子类如需附加结构化元数据(如 Markdown 标题路径),
        应重写本方法或在 `split_text` 之外单独处理。
        """
        out: List[Document] = []
        for doc in docs:
            for i, piece in enumerate(self.split_text(doc.content)):
                meta = dict(doc.metadata)
                meta["chunk"] = i
                meta["splitter"] = self.name
                chunk_doc = Document(content=piece, metadata=meta)
                if chunk_doc:
                    out.append(chunk_doc)
        return out

    @staticmethod
    def _clean(chunks: List[str]) -> List[str]:
        """去空白、丢弃空片段的通用后处理。"""
        return [c.strip() for c in chunks if c and c.strip()]
