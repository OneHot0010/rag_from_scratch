# 阶段2 · 多数据格式接入 + 工程化拆分

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
  splitter.py               # 固定长度切分(带 overlap)
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
逐一实现预留的 PDF / Word / Markdown / HTML loader,并升级向量库为持久化方案(Chroma / Milvus / Qdrant)。
