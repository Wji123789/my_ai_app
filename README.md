# 课程学习助手

> 《大模型应用实训》实验五 综合项目
> 基于 RAG 的课程知识问答 + 学习计划生成

---

## 功能

- **知识问答**：基于《大模型应用实训》实验指导书回答问题，答案可追溯到具体片段
- **诚实拒答**：知识库之外的问题会明确说明「资料不足」，不编造
- **引用来源**：每个回答都可展开查看检索到的原始片段及相似度
- **学习计划**：根据个人情况生成分阶段学习计划，标注重点与易错点
- **自动评估**：内置 12 条测试用例与大模型评分脚本，可量化迭代效果

---

## 快速开始

### 1. 环境准备

需要 Python 3.10+。

```powershell
pip install -r requirements.txt
```

> `sentence-transformers` 会附带安装 PyTorch，体积约 1~2 GB，请预留时间与磁盘空间。

### 2. 配置密钥

**密钥绝不能写进代码。** 从环境变量读取：

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY","sk-你的真实key","User")
```

设置后**必须重开终端**才生效。

### 3. 配置模型镜像（国内必需）

向量模型默认从 huggingface.co 下载，国内无法直连。配置镜像：

```powershell
[Environment]::SetEnvironmentVariable("HF_ENDPOINT","https://hf-mirror.com","User")
[Environment]::SetEnvironmentVariable("HF_HOME","E:\hf_home","User")
```

`HF_HOME` 把模型缓存放到指定目录，避免占用系统盘。

### 4. 构建知识库

`data/knowledge.txt` 是知识文档。修改后需要重建索引：

```powershell
python rebuild_index.py
```

会生成 `data/chunks.json` 与 `data/faiss.index`。

### 5. 启动应用

**必须在终端中运行，不能在 IDLE 里按 F5**（Streamlit 应用是常驻 Web 服务）：

```powershell
streamlit run app.py
```

浏览器会自动打开。停止请按 `Ctrl` + `C`。

---

## 项目结构

```
my_ai_app/
├── app.py                    Streamlit 主程序（界面与流程编排）
├── llm.py                    大模型调用封装（重试 / 异常 / 用量统计）
├── rag.py                    检索与生成逻辑
├── prompts.py                提示词模板加载器
├── evaluate.py               自动评估脚本
├── rebuild_index.py          重建知识库索引
├── test_chunking.py          分块粒度对比实验
├── PRD.md                    产品需求文档
├── requirements.txt
├── .gitignore
│
├── prompts/                  提示词模板库（与代码分离，便于迭代）
│   ├── system_prompt.md        全局角色设定
│   └── task_prompts/
│       ├── rag_qa.md             RAG 问答
│       ├── study_plan.md         学习计划生成
│       └── judge.md              评估打分
│
└── data/
    ├── knowledge.txt         知识文档
    ├── chunks.json           切分后的文本片段
    └── faiss.index           向量索引
```

### 分层说明

| 文件 | 职责 | 改什么来这里 |
|---|---|---|
| `app.py` | 界面与流程 | 调整界面布局、加新页签 |
| `rag.py` | 检索与生成 | 改检索策略、改动 top_k、加过滤逻辑 |
| `llm.py` | 模型调用 | 换平台、改重试策略、改单价 |
| `prompts.py` + `prompts/` | 提示词 | **调提示词只需要改 .md 文件，不用动代码** |

---

## 评估与迭代

```powershell
# 跑一轮评估，结果存为 eval_v1.json
python evaluate.py run v1

# 对比两轮结果
python evaluate.py compare v1 v2

# 查看某轮明细
python evaluate.py show v1
```

评估思路：12 条测试用例覆盖三类情况（核心功能 / 边界情况 / 对抗情况），
用大模型当裁判，从准确性、完整性、相关性三个维度各打 1~5 分。

### 本项目实测结果

| 版本 | 配置 | 综合均分 |
|---|---|---|
| v1（基线） | chunk=300，无目录过滤，top_k=3 | 4.67 |
| v4 | chunk=300，目录过滤，top_k=3 | 4.61 |
| **final** | **chunk=200，目录过滤，top_k=5** | **4.89** |

关键改进：**检索质量，而不是提示词**。
诊断发现「注意事项」类内容的正确片段排在第 21 名，取不到；
通过目录过滤、分块粒度调整、提高 top_k 三步修复。

---

## 常见问题

**Q：提示 `未找到环境变量 DEEPSEEK_API_KEY`**
设置后必须**重开终端**。环境变量的修改对已打开的窗口不生效。

**Q：模型下载卡住**
检查 `HF_ENDPOINT` 是否设置为 `https://hf-mirror.com`。

**Q：`python app.py` 没有任何反应**
Streamlit 应用必须用 `streamlit run app.py` 启动，`python app.py` 不会启动 Web 服务。

**Q：回答总是「资料不足」**
可能是检索没命中。尝试：
- 把侧边栏的 `top_k` 调大
- 换个问法（例如把「实验三中，采集网络数据需要注意哪些问题」改成「采集网络数据的注意事项」）
- 检查 `data/knowledge.txt` 是否包含相关内容

---

## 安全说明

- 密钥通过环境变量读取，`.gitignore` 已排除 `.env` 及相关文件
- 公开部署时应在平台侧配置 Secrets，不要写入代码
- 知识库内容来自公开的实验指导书，仅用于课程学习
