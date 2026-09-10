"""阶段4 · Embedding 模型层单元测试。

每个测试函数独立可运行、打印中间结果,便于 debug。
运行方式:
    PYTHONPATH=. python3 test/test_embedders.py          # 内置 runner(无需 pytest)
    PYTHONPATH=. python3 -m pytest test/test_embedders.py # 有 pytest 时

仅覆盖离线能力(不触发真实 Embedding / API):
    相似度度量 / HashEmbedder 确定性 / 缓存命中与批量去重 / 注册表。
ark:* 依赖 API Key,不在离线单测范围内(见 experiments/compare_embedders.py)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ragcore.models import Document
from ragcore.embedders import (
    build_embedder,
    available_embedders,
    OFFLINE_EMBEDDERS,
    available_metrics,
    get_metric,
    score,
    cosine_scores,
    dot_scores,
    l2_scores,
    HashEmbedder,
    CachedEmbedder,
)


# ---- 相似度度量 ----------------------------------------------------

def test_metrics_registry():
    """注册表列出 cosine/dot/l2;未知度量报错。"""
    names = available_metrics()
    print(f"metrics = {names}")
    for n in ("cosine", "dot", "l2"):
        assert n in names
    try:
        get_metric("nope")
        assert False, "未知度量应报错"
    except ValueError:
        print("metric -> 未知度量已拒绝")


def test_cosine_ignores_magnitude():
    """余弦只看方向:同向不同模长应得分 1;反向应为 -1。"""
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[5.0, 0.0], [-1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    s = cosine_scores(q, m)
    print(f"cosine = {s}")
    assert abs(s[0] - 1.0) < 1e-4
    assert abs(s[1] + 1.0) < 1e-4
    assert abs(s[2] - 0.0) < 1e-4


def test_dot_sensitive_to_magnitude():
    """点积对模长敏感:同向更长者得分更高。"""
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[5.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    s = dot_scores(q, m)
    print(f"dot = {s}")
    assert s[0] > s[1]


def test_l2_returns_negative_distance():
    """L2 返回负距离:越近分越大;完全相同则为 0。"""
    q = np.array([0.0, 0.0], dtype=np.float32)
    m = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)  # 距离 0 和 5
    s = l2_scores(q, m)
    print(f"l2 = {s}")
    assert abs(s[0] - 0.0) < 1e-4
    assert abs(s[1] + 5.0) < 1e-4
    assert s[0] > s[1]  # 越近分越大


def test_score_entry_matches_metric():
    """便捷入口 score(name,...) 应等价于对应度量函数。"""
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    assert np.allclose(score("cosine", q, m), cosine_scores(q, m))
    print("score() 入口与度量函数一致")


# ---- HashEmbedder(离线确定性)-------------------------------------

def test_hash_deterministic_and_shape():
    """相同文本 -> 相同向量;维度符合设定。"""
    emb = HashEmbedder(dim=64)
    v1 = emb.embed_one("机器学习 machine learning")
    v2 = emb.embed_one("机器学习 machine learning")
    print(f"dim={emb.dim}, |v1|={np.linalg.norm(v1):.3f}")
    assert v1.shape == (64,)
    assert np.allclose(v1, v2), "相同文本应得到相同向量(确定性)"


def test_hash_semantic_proximity():
    """共享词的文本余弦相似度应高于无关文本。"""
    emb = HashEmbedder(dim=256)
    vs = np.vstack(emb.embed([
        "机器学习是人工智能的分支",
        "机器学习需要大量数据",
        "今天天气很好适合散步",
    ]))
    q = emb.embed_one("什么是机器学习")
    s = cosine_scores(q, vs)
    print(f"proximity scores = {s}")
    assert s[0] > s[2] and s[1] > s[2], "含相同词者应更相似"


def test_embed_documents_alignment():
    """embed_documents 顺序应与输入 Document 对齐。"""
    emb = HashEmbedder(dim=32)
    docs = [Document(content=f"文本{i}") for i in range(4)]
    vs = emb.embed_documents(docs)
    assert len(vs) == 4
    assert all(v.shape == (32,) for v in vs)
    print("embed_documents 顺序与维度校验通过")


# ---- 缓存(内存,批量去重)-----------------------------------------

def test_cache_hit_on_repeat():
    """第二次向量化相同文本应全部命中缓存。"""
    inner = HashEmbedder(dim=32)
    emb = CachedEmbedder(inner, cache_dir=None)  # 仅内存
    texts = ["a", "b", "c"]
    emb.embed(texts)
    emb.embed(texts)
    st = emb.stats()
    print(f"cache stats = {st}")
    assert st["misses"] == 3, "首轮 3 次未命中"
    assert st["hits"] == 3, "次轮 3 次命中"
    assert st["hit_rate_pct"] == 50.0


def test_cache_dedup_within_batch():
    """同一 batch 内的重复文本只下发底层一次。"""
    class Counting(HashEmbedder):
        name = "counting"

        def __init__(self, dim=16):
            super().__init__(dim=dim)
            self.calls = 0

        def embed(self, texts):
            self.calls += len(texts)
            return super().embed(texts)

    inner = Counting(dim=16)
    emb = CachedEmbedder(inner, cache_dir=None)
    vs = emb.embed(["x", "x", "y", "x", "y"])
    print(f"底层实际计算条数 = {inner.calls}, 返回条数 = {len(vs)}")
    assert len(vs) == 5, "返回条数应与输入一致"
    assert inner.calls == 2, "去重后底层只应算 x、y 两条"
    # 顺序对齐:相同文本得到相同向量
    assert np.allclose(vs[0], vs[1]) and np.allclose(vs[0], vs[3])
    assert np.allclose(vs[2], vs[4])


def test_cache_disk_persist(tmp_path=None):
    """磁盘缓存:新建的第二个 CachedEmbedder 能从磁盘命中。"""
    import tempfile
    d = tempfile.mkdtemp(prefix="emb_cache_test_")
    e1 = CachedEmbedder(HashEmbedder(dim=32), cache_dir=d)
    e1.embed(["persist-me"])
    e2 = CachedEmbedder(HashEmbedder(dim=32), cache_dir=d)
    e2.embed(["persist-me"])
    st = e2.stats()
    print(f"disk persist stats = {st} (dir={d})")
    assert st["hits"] == 1, "第二个实例应从磁盘命中"


# ---- 注册表 --------------------------------------------------------

def test_registry_offline_build():
    """按名字构建 hash-offline;OFFLINE_EMBEDDERS 含之。"""
    names = available_embedders()
    print(f"available = {names}")
    assert "hash-offline" in names
    assert "hash-offline" in OFFLINE_EMBEDDERS
    emb = build_embedder("hash-offline", dim=48)
    assert isinstance(emb, HashEmbedder)
    assert emb.embed_one("t").shape == (48,)


def test_registry_cache_wrapping():
    """cache=True 时应返回 CachedEmbedder 包裹层。"""
    emb = build_embedder("hash-offline", cache=True, cache_dir=None, dim=16)
    assert isinstance(emb, CachedEmbedder)
    assert emb.name == "hash-offline"  # 身份透传
    print("registry -> 缓存包裹与身份透传校验通过")


def test_registry_unknown_raises():
    """未知模型名报错。"""
    try:
        build_embedder("no-such-model")
        assert False, "未知模型应报错"
    except ValueError:
        print("registry -> 未知模型已拒绝")


# ====================================================================
# 运行入口(无 pytest 也能跑)
# ====================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        print("=" * 60)
        print(f"运行: {t.__name__}")
        t()
        passed += 1
    print("=" * 60)
    print(f"全部通过: {passed}/{len(tests)}")
