# rag_from_scratch · 阶段1 最简 Demo(Naive RAG)

从零手写一个最小可问答的 RAG,embedding 与 chat 都接火山(豆包)模型:

`加载 txt → 固定长度切分 → 向量化 → 内存向量库 → Top-K 检索 → 拼 Prompt → LLM 生成`

不依赖 LangChain / LlamaIndex,便于看清 RAG 的完整机理。

## 文件说明
| 文件 | 作用 |
|---|---|
| `rag_stage1_demo.py` | 阶段1 主程序,交互式问答 |
| `sample.txt` | 示例知识库(RAG 基础知识) |
| `requirements.txt` | 依赖清单 |
| `.env` | 火山密钥与模型配置(勿提交到 git) |

## 环境变量(`.env`)
| 变量 | 说明 |
|---|---|
| `VOLCAN_API_KEY` | 火山方舟 API Key(必填) |
| `VOLCAN_BASE_URL` | 火山方舟接入地址,如 `https://ark.cn-beijing.volces.com/api/v3` |
| `EMBED_MODEL` | Embedding 模型,默认 `doubao-embedding-vision-251215` |
| `CHAT_MODEL` | 对话模型,默认 `doubao-seed-2-1-pro-260628` |
| `DOC_PATH` | 问答文档路径,默认 `sample.txt` |

## 运行步骤
1. 建虚拟环境(可选): `python3 -m venv .venv`,再激活(macOS/Linux: `source .venv/活动脚本`,即 activate)
2. 安装依赖: `pip install -r requirements.txt`
3. 配置 `.env`,至少填入 `VOLCAN_API_KEY` 与 `VOLCAN_BASE_URL`
4. 运行: `python rag_stage1_demo.py`

## 链路说明(对应代码)
| 步骤 | 函数 | 说明 |
|---|---|---|
| 加载 | `load_text` | 读取单个 txt 文档 |
| 切分 | `split_text` | 固定长度切分,`chunk_size=300, overlap=50`,重叠防止上下文断裂 |
| 向量化 | `embed` | 调火山 `multimodal_embeddings`,只传 `text` 类型 |
| 检索 | `retrieve` | 内存向量库 + 余弦相似度,取 Top-K(默认 3) |
| 生成 | `generate` | 拼 Prompt 调火山 `chat.completions`,约束"仅依据上下文,无则答『文档中未提及』" |

## 火山模型调用要点
- 客户端: `Ark(api_key=VOLCAN_API_KEY, base_url=VOLCAN_BASE_URL)`
- 向量化: `client.multimodal_embeddings.create(model=EMBED_MODEL, input=[{"type":"text","text":...}])`,取 `resp.data.embedding`
- 对话: `client.chat.completions.create(model=CHAT_MODEL, messages=[...], temperature=0)`,取 `resp.choices[0].message.content`

## 示例问答
输入「RAG 的三个阶段是什么?」后,程序会先打印检索到的 chunk 及余弦相似度,再输出基于上下文的答案。

## 验收标准(Checkpoint)
- 对文档内问题能给出正确答案
- 能打印出被检索到的 chunk 及相似度

## 下一阶段
阶段2「多数据格式接入」:让加载器支持 PDF / Word / Markdown / HTML 等真实格式,并保留元数据(来源、页码)。
