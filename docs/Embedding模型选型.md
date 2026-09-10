# RAG Embedding 模型选型指南

> 阶段4 产出。系统梳理 RAG 索引/检索链路里「向量化(Embedding)」环节的
> 原理、模型谱系与相似度度量,给出可横向对比的实验结论与工程选型建议,
> 并说明批量向量化 + 缓存的收益。
> Embedding 是检索的**语义地基**:切分决定"原料形状",Embedding 决定
> "原料能否在向量空间里被正确地聚到一起"。

---

## 0. 为什么 Embedding 是检索质量的地基

RAG 的检索本质是"在向量空间里找与问题最近的 chunk"。这条链路里:

- **切分(阶段3)** 决定每块内容的边界与完整性;
- **Embedding(阶段4)** 决定这些内容被映射到向量空间后的**相对位置**——
  语义相近的文本是否真的靠得近。

如果 Embedding 模型不理解语义(或语种不匹配),再好的切分与再花哨的检索
策略都是在错误的坐标系里打转。因此**模型选型**直接决定召回质量的上限。

---

## 1. 稠密向量(Dense Vector)原理

**核心思想**:用一个固定维度的浮点向量表示一段文本的语义,使"语义相近 →
向量相近"。

- **稀疏 vs 稠密**:
  - 稀疏向量(如 TF-IDF / BM25)维度=词表大小,大部分为 0,只能匹配
    **字面重合**的词,无法理解"机器学习"与"machine learning"是同义。
  - 稠密向量维度固定(几百~几千),每一维都是学习得到的实数,编码了
    **语义**;"神经网络"和"深度学习"即使不共享字面词,向量也会靠近。
- **怎么来的**:用大规模语料训练的编码器(Transformer 类)把文本压成
  一个稠密向量,训练目标让"语义相近的句子对"向量靠近、"无关句子对"远离
  (对比学习)。
- **本项目落地**:
  - `ArkEmbedder`(`ragcore/embedders/ark.py`)调用火山方舟 multimodal
    embeddings,产出真实语义稠密向量(实测 2048 维)。
  - `HashEmbedder`(`ragcore/embedders/hashing.py`)是**离线确定性**的
    简化实现(哈希词袋 + n-gram + L2 归一化),不理解语义,但满足 embedder
    的全部契约(稳定、可复现、可缓存),用于**无 API Key 时跑通链路、
    做单元测试与教学演示**。⚠️ 召回质量远不及真实模型。

---

## 2. 开源 vs 商用 Embedding

| 维度 | 开源(自部署) | 商用 API |
|---|---|---|
| 代表 | BGE 系列(bge-small/base/large、BGE-M3)、m3e、E5 | OpenAI text-embedding-3、火山方舟 doubao-embedding、Cohere Embed |
| 成本 | 无调用费,但需 GPU/推理资源与运维 | 按量付费,零运维 |
| 数据合规 | 数据不出域,适合敏感/私有数据 | 数据经第三方,需评估合规 |
| 效果 | 头部开源(BGE-M3)已接近商用 | 通常开箱即用、稳定 |
| 可控性 | 可微调、可离线、版本自控 | 依赖厂商,模型可能下线/变更 |
| 起步成本 | 高(部署) | 低(拿 Key 即用) |

**经验法则**:
- 快速起步 / 小规模 / 无运维能力 → **商用 API**(本项目默认走火山方舟)。
- 数据敏感 / 大规模高频调用(成本敏感)/ 需微调 → **开源自部署**(BGE-M3 优先)。

---

## 3. 中英文 / 多语种模型

- **语种匹配至关重要**:纯英文模型处理中文往往召回骤降。中文/中英混合场景
  应选**中文优先或多语种**模型。
- **选型建议**:
  - 中文为主 → **BGE 系列**(bge-large-zh、BGE-M3),中文语义与检索表现好。
  - 多语种 / 跨语言检索(用中文问、召回英文文档)→ **BGE-M3**、
    **multilingual-e5**、OpenAI text-embedding-3(多语种能力强)。
  - 入门 / 资源受限 → bge-small-zh、m3e-base(小而够用)。
  - 生产高质量 → text-embedding-3-large、BGE-M3。
- **本项目**:`HashEmbedder` 的分词对中文按单字 + 2-gram、英文按词处理,
  因此离线基线在中英文上都能演示"共享词更相似";换到 ark 商用模型即可获得
  真正的跨语种语义能力(见第 6 节实验:英文问句同样召回正确英文 chunk)。

---

## 4. 向量维度与相似度度量(cosine / dot / L2)

### 4.1 向量维度

- 维度越高,表达能力通常越强,但**存储/计算成本**与之线性增长;维度过高在
  小数据集上还可能过拟合噪声。
- 维度是模型**固有属性**,选定模型即选定维度(本项目 `ArkEmbedder.dim` 在
  首次调用后回填,实测 2048;`HashEmbedder` 维度可配,默认 256)。
- chunk_size 应与模型最优输入长度匹配:过长会稀释向量语义(见 docs/切分策略.md)。

### 4.2 三种相似度度量

实现见 `ragcore/embedders/similarity.py`。统一约定:所有度量都实现为
`score(query_vec, matrix) -> np.ndarray`,返回值**越大越相似**,便于向量库
直接 `argsort` 取 Top-K。

| 度量 | 语义 | 是否看模长 | 取值 | 适用 |
|---|---|---|---|---|
| **cosine** | 只看方向(夹角) | 否 | [-1, 1],越大越相似 | 文本检索**最常用**,对长度/模长不敏感 |
| **dot(点积)** | 方向 + 模长 | 是 | 无界,越大越相似 | 向量已归一化时≈cosine;未归一化时模长大者更易胜出 |
| **L2(欧氏距离)** | 空间直线距离 | 是 | 距离≥0;本项目返回 **-距离** 以统一"越大越相似" | 距离语义直观,常配合归一化使用 |

> 关键点:**当向量已 L2 归一化时,cosine 与 dot 等价**,且此时 L2 距离与
> cosine 单调对应——三者排序结果趋于一致。这也是第 6 节实验里三种度量
> Recall@k 相同的原因。文本检索默认首选 **cosine**(对模长鲁棒)。

`InMemoryVectorStore` 现已支持通过 `metric=` 配置度量(默认 cosine,
向后兼容阶段1~3 的行为)。

---

## 5. 批量向量化与缓存

Embedding 调用**有耗时与成本**;同一份 chunk 在反复实验、多次建库时会被
重复向量化,浪费 API 额度与时间。`CachedEmbedder`(`ragcore/embedders/cache.py`)
以**装饰器模式**包裹任意 embedder,对上层透明:

- **缓存 key = (模型名, 文本内容) 的 sha256**:模型或文本任一变化即命中不同
  key,换模型天然互不污染。
- **两级缓存**:内存 dict(进程内即时命中)+ 可选磁盘(单条一个 `.npy`,
  跨进程/重启持久化)。
- **批量去重**:一次 `embed` 里重复文本只下发底层模型一次,再按原顺序回填,
  既省调用又保持返回顺序对齐。
- **命中统计**:`stats()` 返回 hits/misses/命中率,供实验量化"缓存收益"。

配置项(见 `ragcore/config.py`):`EMBED_CACHE`(默认开)、`EMBED_CACHE_DIR`
(默认 `.cache/embeddings`,置空则仅内存)。

---

## 6. 对照实验结论

实验脚本:`experiments/compare_embedders.py`。在同一份「8 个 chunk / 8 个
中英文问题(带标准答案)」的迷你评测集上,跑遍多个模型 × 三种度量,输出
**向量维度 / Recall@k / 命中项平均相似度 / 耗时 / 缓存收益**。

以默认评测集(Recall@3)为例:

| 模型 | 维度 | 度量 | Recall@3 | 命中均相似 |
|---|---|---|---|---|
| hash-offline | 256 | cosine | 1.00 | 0.383 |
| hash-offline | 256 | dot | 1.00 | 0.383 |
| hash-offline | 256 | l2 | 1.00 | -1.104 |
| ark:doubao-embedding-vision | 2048 | cosine | 1.00 | **0.651** |
| ark:doubao-embedding-vision | 2048 | dot | 1.00 | 0.650 |
| ark:doubao-embedding-vision | 2048 | l2 | 1.00 | -0.830 |

缓存收益(连续两次向量化同一批 chunk):两模型均 **hits=16 / misses=0 /
命中率 100%**——第二次建库/实验的 Embedding 开销几乎归零。

> 关键观察:
> - **两模型 Recall@3 都达到 1.0**:评测集较小且区分度高,召回率上区分不开;
>   但**命中项平均相似度**上,ark 商用模型(0.651)显著高于离线哈希基线
>   (0.383)——真实语义模型把"正确答案"拉得离问题**更近**,在更大、更含混
>   的语料上这种"排序质量"差异会直接转化为 Recall 差异。
> - **三种度量在归一化向量上排序一致**,故 Recall@3 相同;度量的差异主要
>   体现在分值尺度(l2 为负距离)与是否对模长敏感。
> - **缓存把重复向量化成本降到 0**:实验/多次建库场景收益巨大。
> - **离线哈希模型**在英文问句(如 "which language is used for AI")上也能
>   命中,但靠的是字面词重合而非语义;真实模型才具备跨语种语义泛化能力。

---

## 7. 模型选型:结论与建议

| 场景 | 首选 | 理由 |
|---|---|---|
| 快速起步 / 无运维 / 小规模 | 商用 API(火山方舟 doubao-embedding / OpenAI text-embedding-3) | 拿 Key 即用,效果稳定 |
| 中文为主 | **BGE 系列(bge-large-zh / BGE-M3)** | 中文语义与检索表现好 |
| 多语种 / 跨语言检索 | **BGE-M3** / multilingual-e5 / text-embedding-3 | 跨语种语义泛化强 |
| 入门 / 资源受限自部署 | bge-small-zh / m3e-base | 小而够用,可离线 |
| 生产高质量 | text-embedding-3-large / BGE-M3 | 综合效果与稳定性最佳 |
| 数据敏感 / 高频调用降本 | 开源自部署(BGE-M3) | 数据不出域,规模化后成本更低 |
| 无 API Key / 单元测试 / 演示 | **hash-offline** | 零依赖跑通链路,但仅供演示 |

**度量选型**:文本检索默认 **cosine**;若已对向量做 L2 归一化,cosine/dot/L2
排序等价,选 cosine 即可(对模长最鲁棒)。

**总体经验法则**:不确定就用「商用 API + cosine + 开启缓存」快速起步;当数据
敏感或调用量大到成本敏感时,再迁移到自部署 **BGE-M3**。切模型只需改配置
`EMBED_BACKEND`(或 `ark:<model>`),上层代码不变(开闭原则)。

---

## 8. 复现实验

```bash
# 仅离线基线(无需 API Key)——演示对比骨架
python experiments/compare_embedders.py

# 有 API Key 时:追加一个或多个 ark 商用模型横向对比
python experiments/compare_embedders.py --ark ark:doubao-embedding ark:doubao-embedding-large

# 指定 Recall@k、只比某些度量、关闭缓存
python experiments/compare_embedders.py --k 5 --metrics cosine dot --no-cache

# 离线单元测试(度量 / HashEmbedder / 缓存去重 / 注册表)
PYTHONPATH=. python3 test/test_embedders.py
```

> 未配置 API Key 时,ark 模型会自动跳过,`hash-offline` 基线照常运行。
