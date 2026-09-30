"""
rag.py —— 检索与生成逻辑

职责：
  · 加载向量模型与 FAISS 索引
  · 检索相关文本片段
  · 组装提示词并调用大模型（通过 llm.py，不直接依赖 SDK）

与实验四的区别：
  实验四把逻辑散在 lab4_3.py 里，本模块把它整理成可复用的服务层，
  并把「检索」和「生成」拆开 —— 这样 app.py 和 evaluate.py 都能直接调用，
  评估脚本也能拿到检索结果单独检查检索质量。
"""

import json
import os

import faiss
from sentence_transformers import SentenceTransformer

import llm
import prompts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 模块级缓存：模型和索引只加载一次
_resources = None


def load_resources():
    """加载向量模型、FAISS 索引与文本片段（带缓存）"""
    global _resources
    if _resources is not None:
        return _resources

    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    index_path = os.path.join(DATA_DIR, "faiss.index")

    for p in (chunks_path, index_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少知识库文件：{p}\n"
                "请先运行实验四的 lab4_1.py 和 lab4_2.py 生成，"
                "并把生成的 chunks.json / faiss.index 复制到 data/ 目录。"
            )

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(index_path)
    model = SentenceTransformer(MODEL_NAME)

    _resources = (model, index, chunks)
    return _resources


def _is_toc(text):
    """判断是否为目录型片段。

    目录片段的特征是包含大量 markdown 锚点链接「](#」。
    它们罗列了全文所有小节标题，因此**与任何问题的语义相似度都不低**，
    容易挤占 top_k 名额，把真正有用的正文片段挤出检索结果。

    这个问题在实验四中已观察到，在实验五的评估中被数据证实：
    12 条测试用例中得分最低的 2 条，检索结果里都有目录片段占据前排。
    这是 v2 版本针对性的改进点。
    """
    return text.count("](#") >= 3


def _search_once(query, top_k, min_score):
    """单查询检索：返回 [(片段下标, 相似度), ...]"""
    model, index, chunks = load_resources()

    q_emb = model.encode([query], normalize_embeddings=True)

    # 多取候选：留出被目录片段和低分片段占掉的名额
    fetch = min(index.ntotal, top_k * 4 + 8)
    scores, ids = index.search(q_emb.astype("float32"), fetch)

    out = []
    for i, s in zip(ids[0], scores[0]):
        if i < 0:
            continue
        s = float(s)
        if s < min_score:
            continue
        if _is_toc(chunks[i]):        # 跳过目录片段
            continue
        out.append((int(i), s))
    return out


_REWRITE_PROMPT = """请把下面的问题改写成 {n} 个适合向量检索的查询语句。

要求：
1. 去掉「实验三中」「这门课」「这个实验」这类语境词，只保留核心概念
2. 尽量使用教材、文档里可能出现的**陈述式表述**，而不是疑问句
3. 每行一个查询，不要编号、不要解释、不要引号

问题：{q}
"""


def _rewrite_query(question, n=3):
    """用大模型把用户问题改写成多个检索友好的查询（Query Rewriting）。

    为什么需要：
      用户用疑问句提问（「采集网络数据需要注意哪些问题？」），
      而文档里的答案是以陈述句写的（「注意遵守robots协议与网站条款」）。
      两者在向量空间里距离较远，直接检索容易召回「思考题」这类
      字面相似但并非答案的内容。

    实测证据（本项目的评估中）：
      「实验三中，采集网络数据需要注意哪些问题？」 -> 正确片段排名第 21
      「采集网络数据时需要注意哪些法律与伦理问题？」 -> 同一片段排名第 2
      可见措辞对检索结果影响极大。
    """
    try:
        text = llm.chat(
            [{"role": "user", "content": _REWRITE_PROMPT.format(n=n, q=question)}],
            temperature=0.0,
            max_tokens=200,
        )
    except llm.LLMError:
        return []                     # 改写失败就退回单查询，不影响主流程

    queries = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-·•0123456789.、) ").strip('"“”\'')
        if line and len(line) > 4 and line != question:
            queries.append(line)
    return queries[:n]


def retrieve(question, top_k=3, min_score=0.0, use_rewrite=False):
    """检索最相关的片段。

    返回 [(片段文本, 相似度), ...]

    三道改进（v2）：
      1. 跳过目录型片段 —— 它们罗列全文标题，与任何问题都相似，会挤占名额
      2. 多取候选后再截取 —— 留出被过滤掉的名额
      3. use_rewrite=True 时启用查询改写，多查询检索后合并去重

    相似度低于 min_score 的结果会被丢弃 —— 用于拦截知识库之外的问题。
    """
    model, index, chunks = load_resources()

    # 收集所有候选：(片段下标 -> 最高相似度)
    best = {}
    for i, s in _search_once(question, top_k, min_score):
        best[i] = max(best.get(i, 0.0), s)

    if use_rewrite:
        for q2 in _rewrite_query(question):
            for i, s in _search_once(q2, top_k, min_score):
                best[i] = max(best.get(i, 0.0), s)

    # 按相似度排序，取前 top_k
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
    return [(chunks[i], s) for i, s in ranked]


def answer_question(question, top_k=3, min_score=0.0,
                    temperature=0.1, use_rag=True, use_rewrite=False):
    """RAG 问答。

    use_rag=False      退化为「纯大模型回答」，用于效果对比
    use_rewrite        是否启用查询改写。默认关闭 —— 实测在本项目中
                       改写没有带来改善（真正的问题是分块粒度），
                       而每次问答会多消耗一次 API 调用，因此不默认启用。
    返回 (回答文本, 检索结果列表)
    """
    messages = [{"role": "system", "content": prompts.system_prompt()}]

    if not use_rag:
        messages.append({"role": "user", "content": question})
        return llm.chat(messages, temperature=temperature), []

    hits = retrieve(question, top_k=top_k, min_score=min_score,
                    use_rewrite=use_rewrite)

    # 检索结果为空（或全部低于阈值）——不调用大模型，直接拒答，省 token
    if not hits:
        return ("资料不足：这个问题超出了知识库的范围。"
                "当前知识库只包含《大模型应用实训》实验指导书的内容，"
                "请尝试询问与该课程相关的问题。"), []

    context = "\n\n".join(text for text, _ in hits)
    messages.append({
        "role": "user",
        "content": prompts.rag_qa(context=context, question=question),
    })

    return llm.chat(messages, temperature=temperature), hits


def generate_study_plan(profile, top_k=6):
    """根据学生情况生成学习计划。

    profile 是学生情况描述，例如：
      "我是编程初学者，每周能投入 6 小时，想在 4 周内完成这门课的全部实验"
    """
    # 用学生的情况描述去检索最相关的资料，而不是把整本指导书塞进提示词
    hits = retrieve(profile, top_k=top_k)
    context = "\n\n".join(text for text, _ in hits) if hits else "（无相关资料）"

    messages = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user",
         "content": prompts.study_plan(profile=profile, context=context)},
    ]

    data = llm.chat_json(messages, temperature=0.3, max_tokens=2000)
    return data, hits


if __name__ == "__main__":
    # 简单自测
    model, index, chunks = load_resources()
    print(f"知识库：{len(chunks)} 个片段，索引 {index.ntotal} 个向量")

    q = "实验四的思考题有哪些？"
    ans, hits = answer_question(q)
    print(f"\n问题：{q}")
    print("检索到：")
    for t, s in hits:
        print(f"  {s:.4f}  {t[:50]}…")
    print(f"\n回答：\n{ans}")
    print(f"\n{llm.usage_text()}")
