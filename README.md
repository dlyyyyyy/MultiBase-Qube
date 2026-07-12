<<<<<<< HEAD
# 多源知识库融合问答系统

基于 Streamlit + LangChain + Chroma + BGE + DeepSeek 的多知识库 RAG 问答系统。

## 功能特点

- 📚 **多源知识库**：支持政策库（PDF）、FAQ库（CSV）、工单库（CSV）三个独立知识库
- 🔍 **并行检索**：同时检索三个知识库，每路取 Top 5
- 🔄 **智能重排**：使用 BGE-Reranker 对 15 条结果重排序，精选 Top 3
- 🤖 **深度回答**：基于 DeepSeek API 生成高质量答案
- 📊 **过程透明**：展示检索过程、重排序结果、参考依据
- ⚡ **性能优化**：模型缓存、流式输出、混合检索（向量+BM25）
- 📝 **语义扩展**：同义词词典 + jieba 分词，支持近义词检索

## 技术栈

- **前端框架**：Streamlit
- **RAG 框架**：LangChain
- **向量数据库**：Chroma（本地存储）
- **Embedding 模型**：BAAI/bge-small-zh-v1.5
- **Rerank 模型**：BAAI/bge-reranker-base
- **大模型**：DeepSeek API

## 项目结构

```
.
├── app.py              # Streamlit 前端应用
├── rag_engine.py       # RAG 引擎核心模块
├── config.py           # 配置文件（API Key 从环境变量读取）
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明
├── .gitignore          # Git 忽略规则
├── .streamlit/
│   └── secrets.toml    # Streamlit secrets（本地参考，不提交）
├── data/
│   ├── policies/       # 政策库 PDF 文件
│   ├── faq/            # FAQ 库 CSV 数据
│   └── tickets/        # 工单库 CSV 数据
└── vector_store/       # 向量数据库存储目录（运行时生成，不提交）
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 DeepSeek API Key

API Key 从环境变量读取，**不支持硬编码到代码中**。支持三种配置方式：

**方式一：环境变量（推荐本地开发用）**

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-api-key-here"

# Windows CMD
set DEEPSEEK_API_KEY=your-api-key-here

# Linux/Mac
export DEEPSEEK_API_KEY=your-api-key-here
```

**方式二：.env 文件（本地开发用）**

在项目根目录创建 `.env` 文件（已被 .gitignore 排除）：

```
DEEPSEEK_API_KEY=your-api-key-here
```

**方式三：Streamlit Secrets（Cloud 部署用，见下方部署说明）**

如果未配置 API Key，启动时会报错并提示配置方法。

### 3. 准备数据

将你的数据文件放入对应目录：

- **政策库**：将 PDF 文件放入 `data/policies/` 目录
- **FAQ库**：在 `data/faq.csv` 中填写数据，列名：`question`, `answer`
- **工单库**：在 `data/tickets.csv` 中填写数据，列名：`question`, `answer`

### 4. 构建向量索引

运行应用后，在侧边栏进入「知识库管理」页面，点击「一键构建所有知识库索引」。

或者首次使用时系统会自动加载已有索引。

### 5. 启动应用

```bash
streamlit run app.py
```

## 使用说明

### 智能问答页面

1. 在输入框中输入问题
2. 点击「提交问题」按钮
3. 查看检索过程（三个知识库的检索结果数量）
4. 查看重排序结果
5. 阅读最终答案
6. 查看参考依据（来源 + 章节 + 内容）
7. 可提交问题反馈

### 知识库管理页面

- 分别查看和管理三个知识库
- 支持单独重建某个知识库的索引
- 支持一键构建所有知识库索引

### 数据统计页面

- 查看各知识库数据量
- 查看系统配置信息

### 反馈记录页面

- 查看用户提交的所有反馈记录

## 核心流程

```
用户输入问题
    ↓
同时检索三个知识库（每路 Top 5）
    ↓
合并 15 条结果
    ↓
BGE-Reranker 重排序取 Top 3
    ↓
DeepSeek 生成最终答案
    ↓
展示：检索过程 → 最终答案 → 参考依据
```

## 配置说明

在 `config.py` 中可以调整以下参数：

- `RETRIEVE_TOP_K`：每个知识库检索数量（默认 3）
- `RERANK_TOP_N`：重排序后保留数量（默认 3）
- `CHUNK_SIZE`：PDF 文档切分大小（默认 500）
- `CHUNK_OVERLAP`：切分重叠大小（默认 50）
- `MAX_TOKENS`：大模型输出最大 token 数（默认 512）
- `TEMPERATURE`：生成温度（默认 0.3）
- `ENABLE_HYBRID_SEARCH`：是否启用混合检索（默认 True）
- `ENABLE_SYNONYM_EXPANSION`：是否启用同义词扩展（默认 True）
- `SIMILARITY_THRESHOLD`：向量检索相似度阈值（默认 0.45）

## Streamlit Cloud 部署

### 1. 推送代码到 GitHub

确保代码已推送到 GitHub 仓库。注意 `.gitignore` 已排除以下内容：
- `.env` - 本地环境变量文件
- `.streamlit/secrets.toml` - 本地 secrets 文件
- `vector_store/` - 向量数据库（运行时生成）
- `__pycache__/` - Python 缓存

### 2. 在 Streamlit Cloud 创建应用

1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 点击 "New app"，选择你的 GitHub 仓库
3. 设置主文件路径为 `app.py`
4. 点击 "Deploy"

### 3. 配置 Secrets

在 Streamlit Cloud 的应用管理页面：

1. 进入应用详情页，点击右下角 "Settings" → "Secrets"
2. 添加以下内容：

```toml
DEEPSEEK_API_KEY = "sk-your-real-api-key"
```

3. 点击 "Save"

`config.py` 会自动从 `st.secrets` 读取 API Key。

### 4. 首次启动注意事项

首次部署后，应用需要下载以下模型，可能需要 **5-10 分钟**：

- **BGE Embedding 模型**（BAAI/bge-small-zh-v1.5，约 100MB）
- **BGE Reranker 模型**（BAAI/bge-reranker-base，约 200MB）
- **RapidOCR 模型**（约 50MB）

下载完成后会缓存，后续启动较快。

如果下载超时，可以在 Streamlit Cloud 的 Secrets 中添加 HuggingFace 镜像：

```toml
DEEPSEEK_API_KEY = "sk-your-real-api-key"

HF_ENDPOINT = "https://hf-mirror.com"
```

然后在 `config.py` 中读取此环境变量（已通过 `load_dotenv()` 支持）。

### 5. 构建知识库索引

部署成功后：

1. 在侧边栏进入「知识库管理」页面
2. 上传知识库文件（PDF/CSV）
3. 点击「一键构建所有知识库索引」
4. 索引构建完成后即可正常问答

## 注意事项

1. 首次运行会自动下载 BGE 模型（约 100MB），请确保网络畅通
2. 建议使用 Python 3.9+ 版本
3. 向量索引存储在 `vector_store/` 目录，删除该目录可重建索引
4. 确保 DeepSeek API Key 有效且有足够额度
5. **切勿将 API Key 硬编码到代码中**，始终通过环境变量或 Secrets 配置
6. Streamlit Cloud 首次部署需要等待模型下载，请耐心等待
7. 混合检索依赖 `jieba` 和 `rank-bm25`，已在 requirements.txt 中声明

## 常见问题

**Q: 模型下载慢怎么办？**
A: 可以设置 HuggingFace 镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何更换 Embedding 模型？**
A: 修改 `config.py` 中的 `EMBEDDING_MODEL_NAME` 为其他 BGE 模型即可。

**Q: PDF 加载失败怎么办？**
A: 确保 PDF 文件未加密且可正常读取。复杂格式 PDF 可能需要额外处理。
=======
# MultiBase-Qube
It is a QA agent powered by fused multi-source knowledge bases, capable of extracting valid information from different document libraries to respond to user questions automatically.
>>>>>>> 611c0e749048ac261dc7e712229a666d8473446b
