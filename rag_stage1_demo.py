"""
阶段1 · 最简 Demo(Naive RAG)
端到端跑通一个可问答的最小 RAG:
    加载 txt -> 固定长度切分 -> 向量化 -> 内存向量库 -> Top-K 检索 -> 拼 Prompt -> LLM 生成
刻意"从零手写"每一步(不依赖 LangChain),便于看清 RAG 的完整机理。
依赖: openai, numpy, python-dotenv
"""
import os
import numpy as np
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv


load_dotenv()   
client = Ark(
    api_key=os.getenv("VOLCAN_API_KEY"),
    base_url=os.getenv("VOLCAN_BASE_URL"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "doubao-embedding-vision-251215")
CHAT_MODEL = os.getenv("CHAT_MODEL", "doubao-seed-2-1-pro-260628")


# 1) 加载单个 txt
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 2) 固定长度切分(带 overlap 防止上下文断裂)
def split_text(text, chunk_size=300, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# 3) 文本向量化
def embed(texts):
    res = []
    for text in texts:
        resp = client.multimodal_embeddings.create(
            model=EMBED_MODEL, 
            input=[{"type": "text", "text": text}]
        )
        res.append(np.array(resp.data.embedding, dtype=np.float32))
    return res


# 4/5) 内存向量库 + Top-K 余弦相似检索
def retrieve(query, chunks, chunk_vecs, k=3):
    q = embed([query])[0]
    sims = chunk_vecs @ q / (
        np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(q) + 1e-8
    )
    top = sims.argsort()[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in top]


# 6) 拼 Prompt 调 LLM 生成(约束仅依据上下文,防幻觉)
def generate(query, contexts):
    context_text = "\n\n".join(
        f"[片段{i + 1}] {c}" for i, (c, _) in enumerate(contexts)
    )
    prompt = (
        "你是知识问答助手。仅依据下面的上下文回答问题,"
        "若上下文没有相关信息就回答\"文档中未提及\"。\n\n"
        f"上下文:\n{context_text}\n\n"
        f"问题: {query}\n答案:"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def main():
    doc_path = os.getenv("DOC_PATH", "sample.txt")
    text = load_text(doc_path)
    chunks = split_text(text)
    print(f"文档切分为 {len(chunks)} 个 chunk,正在向量化 ...")
    chunk_vecs = np.array(embed(chunks))
    print("索引构建完成,输入问题开始问答(输入 q 退出)\n")
    while True:
        query = input("问题> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        hits = retrieve(query, chunks, chunk_vecs)
        print("\n--- 检索到的 chunk ---")
        for i, (c, s) in enumerate(hits):
            print(f"[{i + 1}] (相似度 {s:.3f}) {c[:80]} ...")
        answer = generate(query, hits)
        print(f"\n答案: {answer}\n")


if __name__ == "__main__":
    main()
