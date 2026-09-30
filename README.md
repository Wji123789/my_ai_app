# 公考常识学习助手

> 《大模型应用实训》实验五 综合项目
> 基于 RAG 的公考常识问答 + 备考计划生成
> 知识库：《公考常识手册》（政治 / 人文 / 科技 / 法律 / 经济 五类合并为一部手册）

---

## 功能

- **常识问答**：基于《公考常识手册》回答问题，答案可追溯到具体片段
- **诚实拒答**：知识库之外的问题会明确说明「资料不足」，不编造
- **引用来源**：每个回答都可展开查看检索到的原始片段及检索得分
- **分类过滤**：侧边栏可把检索范围锁定在某一分类内，减少串题
- **章节速览**：问「某分类包含哪些内容」时，直接抽取章节目录作答
- **Rerank 重排**：两阶段检索（向量召回 20 条 → 交叉编码器精排），带失败自动降级
- **备考计划**：根据考生基础与备考时间生成分阶段复习计划，标注高频考点与易错点
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

知识库由五份公考常识 PDF 合并为一部手册：

```powershell
python build_knowledge.py     # 五份 PDF → data/knowledge.txt（《公考常识手册》）
python rebuild_index.py       # 切分 + 向量化 + 建 FAISS 索引
```

分别生成 `data/knowledge.txt`、`data/chunks.json`、`data/faiss.index`、`data/parents.json`。

生成的文档结构：

```
公考常识手册
一、政治常识
  【专项一 中共党史】…
二、人文常识
  【专题一 诸子百家】…
三、科技常识  …
四、法律常识  …
五、经济常识  …
```

`build_knowledge.py` 会做表格感知提取（用 PyMuPDF 的 `find_tables()` 恢复表格结构）、
断行修复、上标修复等预处理，把 PDF 还原成可检索的纯文本。

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
├── config.py                 参数配置中心（所有可调参数集中在这里）
├── llm.py                    大模型调用封装（重试 / 异常 / 用量统计）
├── rag.py                    检索与生成逻辑（两阶段检索 / 分类过滤 / 父子分块）
├── prompts.py                提示词模板加载器
├── evaluate.py               自动评估脚本
├── build_knowledge.py        五份 PDF → knowledge.txt
├── rebuild_index.py          切分 + 向量化 + 建索引
├── test_chunking.py          分块粒度对比实验
├── PRD.md                    产品需求文档
├── requirements.txt
├── .gitignore
│
├── prompts/                  提示词模板库（与代码分离，便于迭代）
│   ├── system_prompt.md        全局角色设定
│   └── task_prompts/
│       ├── rag_qa.md             RAG 问答
│       ├── study_plan.md         备考计划生成
│       └── judge.md              评估打分
│
├── testsets/                 测试集（与代码分离）
│   └── gongkao.json             12 条用例，覆盖三类情况
│
└── data/
    ├── knowledge.txt         知识文档（《公考常识手册》）
    ├── chunks.json           切分后的文本片段
    ├── faiss.index           向量索引
    └── parents.json          父子关系（用于上下文补齐）
```

### 分层说明

| 文件 | 职责 | 改什么来这里 |
|---|---|---|
| `app.py` | 界面与流程 | 调整界面布局、加新页签 |
| `config.py` | 参数配置 | 改 top_k、换模型、调价格、开关 Rerank |
| `rag.py` | 检索与生成 | 改检索策略、加过滤逻辑 |
| `llm.py` | 模型调用 | 换平台、改重试策略 |
| `prompts.py` + `prompts/` | 提示词 | **调提示词只需要改 .md 文件，不用动代码** |

---

## 检索流程

```
用户提问
   │
   ├─ 是否「分类总览」类问题？ ── 是 ─→ 抽取该分类章节目录，并入上下文
   │
   ├─ 向量召回候选池（默认 20 条，汇总类问题自动翻倍）
   │     ├─ 跳过目录型片段（含大量 markdown 锚点）
   │     ├─ 按分类过滤（用户可在侧边栏指定）
   │     └─ 按 min_score 拦截
   │
   ├─ Rerank 重排（bge-reranker-base 交叉编码器）
   │     └─ 加载失败自动降级为纯向量检索
   │
   ├─ 取 top_k 片段
   │
   └─ 父子分块：向上补齐相邻片段，恢复被切分点截断的上下文
         │
      拼装提示词 → 调用大模型 → 返回答案 + 引用来源
```

---

## 评估与迭代

```powershell
# 跑一轮评估，结果存为 eval_v6.json
python evaluate.py run v6 --rerank

# 关闭 Rerank 跑一轮，用于对比
python evaluate.py run v6_base --no-rerank

# 对比两轮结果
python evaluate.py compare v6_base v6

# 查看某轮明细
python evaluate.py show v6
```

评估思路：12 条测试用例覆盖三类情况（核心功能 / 边界情况 / 对抗情况），
用大模型当裁判，从准确性、完整性、相关性三个维度各打 1~5 分。
测试集存放在 `testsets/gongkao.json`，换套题不用改代码。

### 本项目实测结果

**更换知识库前的旧知识库（《大模型应用实训》指导书）：**

| 版本 | 配置 | 综合均分 |
|---|---|---|
| v1（基线） | chunk=300，无目录过滤，top_k=3 | 4.67 |
| v4 | chunk=300，目录过滤，top_k=3 | 4.61 |
| final | chunk=200，目录过滤，top_k=5 | 4.89 |

**更换为公考常识知识库后：**

| 版本 | 配置 | 综合均分 |
|---|---|---|
| gk_base | 纯向量，top_k=5 | 4.75 |
| gk_rerank | +Rerank 重排，top_k=5 | 4.75 |
| **gk_v7** | **五类合并为一部手册 + 全部修复，启用重排** | **5.00** |

前两轮综合均分相同，但 **检索得分（top_score）差异显著**：
核心功能题从 0.55~0.80 提升到 0.93~1.00，说明重排确实让正确片段排到了第一位。
综合均分没变，是因为裁判打分已在天花板附近（12 题中 11 题满分）。

### 一次典型的诊断过程

第 12 题「法律常识都包含哪些内容」在前两轮都是唯一失分题（2.00 分）。
排查发现**不是"检索不准"，而是结构问题**：

1. 命中了一个**空壳片段**——分节标题被切成独立的 22 字符片段，
   字面上命中问题里每个词，排第一却不含任何内容。
2. 法律分类有 **97 个片段**，答案（民法+刑法+宪法三个专题名）分散在其中。
   FAISS top-k 只能召回局部片段，**扩大候选池、加重排都无法让池外内容进来**。

修复方式是按问题类型分流：

| 问题 | 修复 |
|---|---|
| 空壳片段抢占名额 | 建索引时剔除（`is_skeleton()`） |
| 条目被切分点腰斩 | 切分前给「1.」「一、」「【专题…」插空行，提升条目边界优先级 |
| 分类总览类问题 | 抽出章节目录，不走向量检索（`_section_outline()`） |
| 分类标注失准 | 顺延法 + 关键词校验（`_annotate_modules()`） |

**另外记录一次失败的尝试**：曾给重排叠加「上下文得分」（把片段与后一片
拼接后再打分，按 0.5/0.5 融合），结果反而变差 —— 排在答案前一位的无关片段
因为它的「后一片」正好是答案，上下文分拿到 0.9994，被顶到了前排。
**把错误答案推了上去**。已回滚，并在代码里留下说明：
碎片截断应在**切分阶段**解决，而不是在排序阶段打补丁。

---

## 常见问题

**Q：提示 `未找到环境变量 DEEPSEEK_API_KEY`**
设置后必须**重开终端**。环境变量的修改对已打开的窗口不生效。

**Q：模型下载卡住**
检查 `HF_ENDPOINT` 是否设置为 `https://hf-mirror.com`。

**Q：部署后提示 Rerank 模型加载失败**
免费部署环境内存通常只有 1GB，而 reranker 模型约 1GB。
应用会自动降级为纯向量检索，功能不受影响。也可以在环境变量里设 `RERANK_ENABLED=0` 显式关闭。

**Q：`python app.py` 没有任何反应**
Streamlit 应用必须用 `streamlit run app.py` 启动，`python app.py` 不会启动 Web 服务。

**Q：回答总是「资料不足」**
可能是检索没命中。尝试：
- 把侧边栏的 `top_k` 调大
- 换个问法（例如把「法律常识都包含哪些内容」改成「民法包含哪些内容」）
- 检查 `data/knowledge.txt` 是否包含相关内容

---

## 安全说明

- 密钥通过环境变量读取，`.gitignore` 已排除 `.env` 及相关文件
- 公开部署时应在平台侧配置 Secrets，不要写入代码
- 知识库内容来自公开出版的公考常识手册，仅用于学习用途
