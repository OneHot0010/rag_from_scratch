# 阶段2/3 · 多格式接入 · 工程化拆分 · 切分策略对比

在阶段1(单文件 Naive RAG)基础上完成两件事:

1. **数据格式抽象为接口**:每种格式一个 `DataLoader` 子类。本阶段实现
   **Excel / CSV**(及 text),其余 PDF / Word / Markdown / HTML 仅预留接口
   (调用即抛 `NotImplementedError`),便于后续渐进扩展。
2. **按功能拆分模块**:原 `demo.py` 的加载/切分/向量化/检索/生成拆到
   `ragcore/` 各文件,职责单一、便于维护与测试。

## 目录结构
```
main.py                     # 交互式入口(支持多文件、按扩展名自动分派)
demo.py                     # 阶段1 原始单文件版本(保留对照)
ragcore/
  config.py                 # 集中配置(.env / 环境变量)
  models.py                 # Document 数据模型(content + metadata 溯源)
  splitters/                # 切分策略族(阶段3,可插拔 + 可对比)
    base.py                 # Splitter 抽象接口(split_text / split_documents)
    fixed.py                # 固定长度(带 overlap)——最简兜底
    recursive.py            # 递归字符切分——通用默认,保住语义边界
    sentence.py             # 句子 / 段落切分 + 贪心合并
    markdown.py             # 结构感知,注入 heading_path 元数据
    semantic.py             # 语义切分(句间 embedding 断点,需 embedder)
    registry.py             # 按策略名分派的注册表
  embedder.py               # 文本向量化(Ark embeddings)
  vector_store.py           # 内存向量库 + Top-K 余弦检索
  generator.py              # 拼 Prompt 调 LLM 生成(防幻觉)
  pipeline.py               # RAG 编排:加载→切分→向量化→检索→生成
  loaders/
    base.py                 # DataLoader 抽象接口 + 数据模型契约
    text_loader.py          # .txt / .text        [已实现]
    csv_loader.py           # .csv(标准库)       [已实现]
    excel_loader.py         # .xlsx / .xlsm(openpyxl) [已实现]
    stub_loaders.py         # PDF/Word/Markdown/HTML  [仅接口]
    registry.py             # 按扩展名分派的注册表
```

## 阶段3 · 切分策略对比

把「切分(Chunking)」从单一固定长度,抽象为**可插拔、可横向对比**的策略族
(与 loaders 同构:新增策略=新增子类+注册,不改核心流程)。

| 策略 | 名称 | 思路 | 需 API |
|---|---|---|---|
| 固定长度 | `fixed` | 定长滑窗 + overlap,最简兜底 | 否 |
| 递归字符 | `recursive` | 按分隔符优先级递归切,块最均匀——**默认** | 否 |
| 句子 | `sentence` | 按句末标点切句再贪心合并 | 否 |
| 段落 | `paragraph` | 按空行分段,超长回退句子切分 | 否 |
| Markdown | `markdown` | 按标题层级切,注入 `heading_path` 锚点 | 否 |
| 语义 | `semantic` | 句间 embedding 相似度骤降处断开 | 是 |

- 切分策略由 `SPLIT_STRATEGY`(默认 `recursive`)配置,或运行时 `--strategy` 覆盖。
- **对照实验**:`python experiments/compare_splitters.py`,输出各策略的
  chunk 数 / 长度分布 / 长度标准差 / 样例,便于选型。
- **选型与调参建议**:见 [`docs/切分策略.md`](docs/切分策略.md)。

## 接口设计要点
- `DataLoader`(`loaders/base.py`)统一契约:声明 `extensions`,实现
  `load(path) -> List[Document]`。
- `Document`(`models.py`)携带 `content` 与 `metadata`(source / sheet /
  row / chunk 等),使检索结果可溯源。
- `LoaderRegistry`(`loaders/registry.py`)按扩展名分派;**新增格式只需
  实现子类并注册,不改动核心流程**(开闭原则)。

## 运行步骤
1. 安装依赖: `pip install -r requirements.txt`
2. 配置密钥: `cp .env.example .env`,填入 `VOLCAN_API_KEY`(内部网关填 `VOLCAN_BASE_URL`)
3. 运行:
   ```bash
   python main.py                 # 用 .env 里的 DOC_PATH
   python main.py sample.csv      # 指定 CSV
   python main.py a.txt b.xlsx    # 多文件混合接入
   python main.py doc.md --strategy markdown   # 指定切分策略
   ```

   对照切分策略并生成指标表:
   ```bash
   python experiments/compare_splitters.py            # 默认 sample.txt
   python experiments/compare_splitters.py doc.md --size 300 --overlap 50
   ```

   跑单元测试(离线策略,无需 API):
   ```bash
   PYTHONPATH=. python3 test/test_splitters.py
   ```

## 已支持格式
| 格式 | 扩展名 | 状态 |
|---|---|---|
| 文本 | .txt / .text | ✅ 已实现 |
| CSV | .csv | ✅ 已实现 |
| Excel | .xlsx / .xlsm | ✅ 已实现 |
| PDF | .pdf | 🔲 仅接口 |
| Word | .docx / .doc | 🔲 仅接口 |
| Markdown | .md / .markdown | 🔲 仅接口 |
| HTML | .html / .htm | 🔲 仅接口 |

## 下一阶段
升级向量库为持久化方案(Chroma / Milvus / Qdrant);引入混合检索(BM25 + 向量 + RRF 融合)与 Rerank 重排。
