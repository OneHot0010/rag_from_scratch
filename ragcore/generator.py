"""生成模块:拼 Prompt 调 LLM 生成(约束仅依据上下文,防幻觉)。"""
from __future__ import annotations

from typing import List, Tuple

from volcenginesdkarkruntime import Ark

from .config import settings
from .models import Document


class Generator:
    def __init__(self, client: Ark | None = None, model: str | None = None) -> None:
        self.client = client or Ark(api_key=settings.api_key, base_url=settings.base_url)
        self.model = model or settings.chat_model

    def generate(self, query: str, contexts: List[Tuple[Document, float]]) -> str:
        context_text = "\n\n".join(
            f"[片段{i + 1}] {doc.content}" for i, (doc, _) in enumerate(contexts)
        )
        prompt = (
            "你是知识问答助手。仅依据下面的上下文回答问题,"
            "若上下文没有相关信息就回答\"文档中未提及\"。\n\n"
            f"上下文:\n{context_text}\n\n"
            f"问题: {query}\n答案:"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
