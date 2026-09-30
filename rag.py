"""
rag.py —— 检索与生成逻辑

职责：
  · 加载向量模型、Rerank 模型与 FAISS 索引
  · 检索相关文本片段（含两阶段检索：向量召回 + 交叉编码器重排）
  · 组装提示词并调用大模型（通过 llm.py，不直接依赖 SDK）

与实验四的区别：
  实验四把逻辑散在 lab4_3.py 里，本模块把它整理成可复用的服务层，
  并把「检索」和「生成」拆开 —— 这样 app.py 和 evaluate.py 都能直接调用，
  评估脚本也能拿到检索结果单独检查检索质量。

实验五新增的三项检索改进：
  1. 两阶段检索（v3）：FAISS 召回 RERANK_CANDIDATES 条候选，
     再用 bge-reranker-base 交叉编码器精排，取前 top_k。
     向量检索是「双塔」结构，问题和文档各自编码、互不知情；
     交叉编码器把问题和文档拼在一起过一遍模型，能识别否定、
     数量、条件等细粒度差异。实测 20 条候选重排仅需约 2 秒。
     在 Streamlit Cloud 的 1GB 内存上限下有 OOM 风险，
     因此设计为可开关，且加载失败会自动降级为纯向量检索。
  2. 目录片段过滤（v2）：含大量 markdown 锚点的片段与任何问题
     相似度都不低，会挤占 top_k 名额。
  3. 模块过滤：按常识模块（政治/法律/科技/经济/人文）限定检索范围，
     用户明确知道要问哪个模块时能显著减少串题。
"""

import functools
import json
import logging
import os
import re

import faiss
from sentence_transformers import SentenceTransformer

import config
import llm
import prompts

logger = logging.getLogger("rag")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_NAME = config.EMBEDDING_MODEL
RERANKER_NAME = config.RERANKER_MODEL

# 模块级缓存：模型和索引只加载一次
_resources = None
_reranker = None
_reranker_failed = False

# 汇总类问题的意图词。
# 「法律模块都包含哪些内容」「民法典有哪些要点」这类问题不指向某个具体条目，
# 而是在问「这一类里有什么」，答案天然分散在多个章节。
# 实测：常规 20 条候选池里，答案通常在 60 名开外 —— 扩大池子是必要的。
_AGG_RE = re.compile(
    r"(包含|包括|有哪些|有什么|哪些内容|都涉及|涵盖|分别是什么|"
    r"汇总|归纳|总结一下|梳理|分类|种类|类型|都考)"
)

# 「模块总览」类问题：「法律模块都包含哪些内容」「XX 部分有哪些考点」。
# 这类问题的标准答案是**该模块的章节目录**，而目录信息在向量空间里
# 并不是某几个片段的语义邻居 —— 它是一条横跨全模块的索引信息。
# 实测：法律模块相关片段有 122 个，top_k=5 无论怎么排都覆盖不到
# 「民法 + 刑法 + 宪法」三个专题名。这是 FAISS top-k 的结构性限制，
# 扩大候选池、重排序都只能重排池内候选，无法让池外内容进来。
# 因此对这类问题走「章节速览」通道：直接从原文抽取该模块的小节标题。
_MODULE_RE = re.compile(r"(政治|法律|科技|经济|人文)\s*(常识|模块|部分)")


def _detect_module(question, modules):
    """从问题中识别用户提到的模块名"""
    m = _MODULE_RE.search(question)
    if m:
        return m.group(1)
    # 退一步：问题里直接出现了模块名（如「我想复习法律」）
    for mod in modules:
        if mod in question:
            return mod
    return ""


def _is_module_overview(question, modules):
    """判断是否为「模块总览」类问题：既提到模块，又在问包含什么"""
    return bool(_detect_module(question, modules)) and bool(
        re.search(r"(包含|包括|有哪些|有什么|哪些内容|都涉及|涵盖|"
                  r"梳理|分类|类型|考点|框架|目录)", question))


def _section_outline(text, module):
    """从原文里抽取某细分小类的章节目录。

    不依赖向量检索 —— 这是「结构信息」而非「语义相似」问题。
    knowledge.txt 的层级是：
        公考常识手册
        一、政治常识
        二、人文常识
        …
        四、法律常识
        【专题一 民法】
        一、自然人的民事权利能力和民事行为能力
        二、宣告失踪和宣告死亡
        ...
    """
    lines = text.split("\n")
    out, hit = [], False

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 定位到目标小类的开头（「四、法律常识」）
        if not hit:
            if re.match(rf"^[一二三四五六七八九十]+、\s*{re.escape(module)}常识$", s):
                hit = True
            continue

        # 遇到下一个小类就停止
        if re.match(r"^[一二三四五六七八九十]+、\s*\S+常识$", s):
            break

        # 专题标题：【专题一 民法】
        if re.match(r"^【(专题|专项)[^】]*】$", s):
            out.append(f"■ {s.strip('【】')}")
            continue

        # 小节：一、xxx / 二、xxx
        # 注意：小类标题也是「X、」格式，靠上面的 break 已经拦住了
        if re.match(r"^[一二三四五六七八九十]+、", s) and len(s) < 30:
            out.append(f"  {s}")

    return out


def _is_aggregation(question):
    """判断是否为「跨章节汇总」类问题"""
    return bool(_AGG_RE.search(question))


def _module_of(text):
    """从片段文本推断它属于哪个细分小类。

    知识库已合并为单一的「公考常识手册」，
    其下用「一、政治常识」「二、人文常识」这类小节划分细分小类。
    片段只有紧跟在小节标题之后时才会带上它，其余靠顺延法推断。
    """
    head = text[:60]
    for m in config.MODULES:
        # 新结构：一、政治常识
        if re.search(rf"[一二三四五]、\s*{m}常识", head):
            return m
        # 兼容旧结构：第一部分　政治常识 / 政治常识（来源：…）
        if f"{m}常识" in head:
            return m
    return ""


def _infer_module_by_keyword(text):
    """按关键词兜底推断模块（仅在顺延法失效时使用）"""
    scores = {}
    for m, kws in _MODULE_KEYWORDS.items():
        scores[m] = sum(text.count(k) for k in kws)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ""


# 各模块的内容关键词，用于在标题缺失时兜底判断归属
_MODULE_KEYWORDS = {
    "政治": ["党", "社会主义", "革命", "人大", "国体", "政体", "外交", "发展理念"],
    "人文": ["儒家", "道家", "孔子", "孟子", "节气", "节日", "唐诗", "宋词",
             "诸子", "科举", "八大家"],
    "科技": ["物理", "化学", "生物", "细胞", "光", "声", "力", "气体", "金属",
             "天文", "地理", "计算机"],
    "法律": ["民法", "刑法", "宪法", "合同", "侵权", "犯罪", "诉讼", "刑罚",
             "民事", "刑事", "权利", "义务", "时效"],
    "经济": ["经济", "市场", "价格", "需求", "供给", "GDP", "财政", "货币",
             "系数", "税", "通货膨胀"],
}


def _annotate_modules(chunks):
    """给每个片段补上模块标记。

    主路径是「顺延法」：knowledge.txt 里各模块按顺序排列，
    切分后顺序不变，所以在没有新标题出现时沿用上一个模块。
    但顺延法在「片段跨模块边界」或「首个片段不带标题」时会失准，
    因此对推断结果做一次关键词校验 —— 严重不符时改用关键词判定。
    """
    current = ""
    mods = []
    for c in chunks:
        found = _module_of(c)
        if found:
            current = found
            mods.append(current)
            continue

        if current == "":
            # 还没碰到任何模块标题（例如开头就是正文），按关键词判断
            current = _infer_module_by_keyword(c)
            mods.append(current)
            continue

        # 校验：片段里出现的模块关键词是否与顺延结果一致
        by_kw = _infer_module_by_keyword(c)
        if by_kw and by_kw != current:
            # 关键词指向别的模块 —— 以关键词为准（顺延法在这里失准了）
            current = by_kw
        mods.append(current)

    return mods


def load_resources():
    """加载向量模型、FAISS 索引、文本片段与父子关系（带缓存）"""
    global _resources
    if _resources is not None:
        return _resources

    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    index_path = os.path.join(DATA_DIR, "faiss.index")
    parents_path = os.path.join(DATA_DIR, "parents.json")

    for p in (chunks_path, index_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少知识库文件：{p}\n"
                "请先运行 build_knowledge.py 生成 knowledge.txt，"
                "再运行 rebuild_index.py 生成 chunks.json / faiss.index。"
            )

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(index_path)
    model = SentenceTransformer(MODEL_NAME)
    mods = _annotate_modules(chunks)

    # 父子关系缺失时退化为「无上下文补齐」，不影响检索本身
    parents = None
    if os.path.exists(parents_path):
        with open(parents_path, encoding="utf-8") as f:
            parents = json.load(f)

    logger.info("知识库加载完成：%d 个片段，索引 %d 个向量，父子关系 %s",
                len(chunks), index.ntotal,
                "已加载" if parents else "缺失（已降级）")

    _resources = (model, index, chunks, mods, parents)
    return _resources


def load_reranker():
    """按需加载 Rerank 模型。

    三种情况都返回 None，让调用方降级为纯向量检索：
      · 配置里关闭了 rerank
      · 依赖库缺失（老环境没装 sentence-transformers 的 CrossEncoder 分支）
      · 模型下载 / 加载失败（Streamlit Cloud 1GB 内存下可能 OOM）
    """
    global _reranker, _reranker_failed

    if not config.RERANK_ENABLED:
        return None
    if _reranker_failed:
        return None
    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_NAME)
        logger.info("Rerank 模型加载完成：%s", RERANKER_NAME)
    except Exception as e:                      # noqa: BLE001 —— 任何异常都降级
        logger.warning("Rerank 模型加载失败，降级为纯向量检索：%s", e)
        _reranker_failed = True
        _reranker = None

    return _reranker


def rerank_enabled():
    """供界面显示当前检索模式"""
    return config.RERANK_ENABLED and not _reranker_failed


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


def _search_once(query, top_k, min_score, module=""):
    """单查询向量召回：返回 [(片段下标, 相似度), ...]"""
    model, index, chunks, mods, parents = load_resources()

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
        if module and mods[i] != module:   # 限定模块
            continue
        out.append((int(i), s))
    return out


def _apply_rerank(question, candidates, top_k):
    """用交叉编码器对候选重新打分排序。

    candidates: [(片段下标, 向量相似度), ...]
    返回 [(片段下标, 重排分), ...]

    关键实测结论：**交叉编码器的「自身分」是最可靠的排序依据**。

    曾经试过给重排叠一层「上下文得分」（把片段和后一片拼起来再打分，
    再按 0.5/0.5 融合），目的是解决碎片截断问题。结果反而变差：
    排在答案前一位的无关片段，因为它的「后一片」正好是答案，
    上下文分拿到 0.9994，融合后被顶到了前排 —— **把一个错误答案推了上去**。
    而正确答案本身的自身分就是 0.9990，不需要任何补救。

    所以这里只按自身分排序。碎片截断的问题应当在**切分阶段**解决
    （见 rebuild_index.py 的条目边界预处理），而不是在排序阶段打补丁。

    加载失败或调用异常时原样返回候选（截前 top_k），保证检索不中断。
    """
    model = load_reranker()
    if model is None or not candidates:
        return candidates[:top_k]

    _, _, chunks, _, _parents = load_resources()
    try:
        scores = model.predict([(question, chunks[i]) for i, _ in candidates])
    except Exception as e:                      # noqa: BLE001
        logger.warning("Rerank 打分失败，退回向量相似度：%s", e)
        return candidates[:top_k]

    rescored = [(i, float(sc)) for (i, _), sc in zip(candidates, scores)]
    rescored.sort(key=lambda kv: -kv[1])
    return rescored[:top_k]


_REWRITE_PROMPT = """请把下面的问题改写成 {n} 个适合向量检索的查询语句。

要求：
1. 去掉「这门课」「这个实验」「这套教材」这类语境词，只保留核心概念
2. 尽量使用教材、文档里可能出现的**陈述式表述**，而不是疑问句
3. 每行一个查询，不要编号、不要解释、不要引号

问题：{q}
"""


def _rewrite_query(question, n=3):
    """用大模型把用户问题改写成多个检索友好的查询（Query Rewriting）。

    为什么需要：
      用户用疑问句提问（「正当防卫要满足什么条件？」），
      而资料里的答案是名词短语式的条目（「行为｜构成条件｜处罚 | 正当防卫 |
      1.不法侵害正在进行…」）。两者在向量空间里距离较远。

    实测证据（本项目评估中）：
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


def _expand_context(idx, chunks, parents):
    """父子分块：把命中片段与其前一片段拼接，恢复被切分点截断的上下文。

    RAG 的经典矛盾是「小块检索准、大块上下文全」：
      · 块太大 —— 向量被稀释，检索排名靠后
      · 块太小 —— 答案的上下文被腰斩，大模型答不全

    父子分块取两者之长：用 200 字符的小块做精确匹配，
    命中后向上补齐相邻片段，把连续上下文还给大模型。

    受 PARENT_CHAR_BUDGET 约束，避免拼接后超出预期长度。
    """
    if not config.PARENT_CHILD_ENABLED or not parents:
        return chunks[idx]

    budget = config.PARENT_CHAR_BUDGET
    parts = [chunks[idx]]

    # 向前补齐：优先保留「命中片段之前」的内容
    for k in range(1, config.PARENT_MAX_EXPAND + 1):
        j = idx - k
        if j < 0:
            break
        prev = chunks[j]
        if sum(len(p) for p in parts) + len(prev) > budget:
            break
        parts.insert(0, prev)

    return "\n".join(parts)


def retrieve(question, top_k=None, min_score=None, use_rewrite=False,
             module="", use_rerank=None):
    """检索最相关的片段。

    返回 [(片段文本, 分数, 片段下标), ...]

    流程：
      1. FAISS 向量召回候选池
         （跳过目录片段、按模块过滤、按 min_score 拦截）
         汇总类问题自动扩大候选池
      2. 若启用则用交叉编码器重排，取前 top_k
      3. 父子分块：把命中片段向上补齐上下文

    use_rerank=None 表示跟随全局配置。
    相似度低于 min_score 的结果会被丢弃 —— 用于拦截知识库之外的问题。
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    if min_score is None:
        min_score = config.DEFAULT_MIN_SCORE
    if use_rerank is None:
        use_rerank = config.RERANK_ENABLED

    _, _, chunks, _, parents = load_resources()

    # 候选数量：重排需要更宽的候选池；不重排时取 4 倍 top_k 即可
    pool = config.RERANK_CANDIDATES if use_rerank else top_k * 4

    # 汇总类问题：答案分散在多个章节，扩大召回提高覆盖概率
    if _is_aggregation(question):
        pool *= config.AGGREGATION_POOL_MULTIPLIER
        logger.info("检测到汇总类问题，候选池扩大至 %d", pool)

    pool = min(pool, len(chunks))

    # 收集所有候选：(片段下标 -> 最高相似度)
    best = {}
    for i, s in _search_once(question, pool, min_score, module=module):
        best[i] = max(best.get(i, 0.0), s)

    if use_rewrite:
        for q2 in _rewrite_query(question):
            for i, s in _search_once(q2, pool, min_score, module=module):
                best[i] = max(best.get(i, 0.0), s)

    # 按向量相似度排序，作为重排的输入候选池
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:pool]

    if use_rerank:
        ranked = _apply_rerank(question, ranked, top_k)
    else:
        ranked = ranked[:top_k]

    # 父子分块：注意保留原始片段下标，供评估脚本定位
    return [(_expand_context(i, chunks, parents), s, i) for i, s in ranked]


def modules_present():
    """返回知识库中实际存在的模块列表，用于界面下拉框"""
    _, _, _, mods, _ = load_resources()
    seen = []
    for m in mods:
        if m and m not in seen:
            seen.append(m)
    return seen


def _outline_for(question, module):
    """若问题是「模块总览」类的，返回该模块的章节目录文本。

    这是对「结构信息类问题」的专门处理。向量检索解决的是「语义相似」，
    但「法律模块包含哪些内容」问的是「这个模块的目录是什么」——
    目录信息不是任何单个片段的语义邻居，它横跨全模块。
    只在文本层面抽取即可，无需检索。

    未命中时返回 ""，主流程照常走检索。
    """
    _, _, _, mods, _ = load_resources()
    target = module or _detect_module(question, mods)
    if not target or not _is_module_overview(question, mods):
        return ""

    knowledge_path = os.path.join(DATA_DIR, "knowledge.txt")
    if not os.path.exists(knowledge_path):
        return ""

    with open(knowledge_path, encoding="utf-8") as f:
        outline = _section_outline(f.read(), target)

    if not outline:
        return ""

    logger.info("命中「模块总览」通道：%s（%d 个条目）", target, len(outline))
    return f"【{target}常识 · 章节目录】\n" + "\n".join(outline)


def answer_question(question, top_k=None, min_score=None,
                    temperature=0.1, use_rag=True, use_rewrite=False,
                    module="", use_rerank=None):
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

    # 「模块总览」类问题：先取章节目录，作为补充材料并入上下文。
    # 这类答案在向量检索里天然拼不全（见 _section_outline 的说明）。
    outline = _outline_for(question, module)

    hits = retrieve(question, top_k=top_k, min_score=min_score,
                    use_rewrite=use_rewrite, module=module,
                    use_rerank=use_rerank)

    # 检索结果为空（或全部低于阈值）
    # 但有章节目录时不算空 —— 目录本身就是这类问题的答案
    if not hits and not outline:
        return ("资料不足：这个问题超出了知识库的范围。"
                "当前知识库只包含公考常识（政治、法律、科技、经济、人文）的内容，"
                "请尝试询问与该范围相关的问题。"), []

    parts = []
    if outline:
        parts.append(outline)
    parts.extend(text for text, _, _ in hits)

    messages.append({
        "role": "user",
        "content": prompts.rag_qa(context="\n\n".join(parts),
                                  question=question),
    })

    return llm.chat(messages, temperature=temperature), hits


def generate_study_plan(profile, top_k=6, module=""):
    """根据考生情况生成备考计划。

    profile 是考生情况描述，例如：
      "我是应届生，行测常识部分很弱，每天能投入 1 小时，想在 4 周内
       把政治和法律两大块过一遍"
    """
    # 用考生的描述去检索最相关的资料，而不是把整份常识手册塞进提示词
    hits = retrieve(profile, top_k=top_k, module=module)
    context = "\n\n".join(text for text, _, _ in hits) if hits else "（无相关资料）"

    messages = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user",
         "content": prompts.study_plan(profile=profile, context=context)},
    ]

    data = llm.chat_json(messages, temperature=0.3, max_tokens=2000)
    return data, hits


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="rag.py 自测")
    ap.add_argument("question", nargs="?",
                    default="正当防卫的构成条件是什么？")
    ap.add_argument("--no-rerank", action="store_true", help="关闭重排")
    ap.add_argument("--module", default="", help="限定模块")
    args = ap.parse_args()

    model, index, chunks, mods, parents = load_resources()
    print(f"知识库：{len(chunks)} 个片段，索引 {index.ntotal} 个向量")
    print(f"模块：{'、'.join(modules_present())}")
    print(f"重排：{'开启' if not args.no_rerank else '关闭'}\n")

    with_rerank = retrieve(args.question, use_rerank=not args.no_rerank,
                           module=args.module)
    without = retrieve(args.question, use_rerank=False, module=args.module)

    print(f"问题：{args.question}\n")
    print("【纯向量检索 top5】")
    for t, s, i in without:
        print(f"  #{i:<4} {s:.4f}  {t[:44]}")
    print("\n【Rerank 重排后 top5】")
    for t, s, i in with_rerank:
        print(f"  #{i:<4} {s:>8.4f}  {t[:44]}")

    ans, hits = answer_question(args.question, module=args.module)
    print(f"\n【回答】\n{ans}")
    print(f"\n{llm.usage_text()}")
