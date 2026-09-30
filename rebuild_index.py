"""
rebuild_index.py —— 重建知识库索引

背景：
    实验四使用 chunk_size=300 切分公考常识类长文档时，一条常识条目
    （如「二十四节气」「唐宋八大家」）常被拆到两个片段里，
    检索排名靠后甚至取不到。

    在实验五的分块粒度对比实验中（目标片段「robots」的检索排名，旧知识库）：
        chunk_size=300 -> 第 21 名
        chunk_size=200 -> 第 4 名    ← 本项目采用
        chunk_size=150 -> 第 3 名
        chunk_size=120 -> 第 3 名

    更细的分块提升了检索精度，但也会削弱上下文完整性，
    因此不宜过细 —— 实测 chunk_size=150 时多项用例反而退化。
    折中取 200，与 config.py 保持一致。

本版新增的两项结构性处理（针对「跨章节汇总」类问题）：
  ① 过滤空壳片段
     分节标题「第四部分　法律常识（来源：法律常识.pdf）」会被切成一个
     只有 22 字符的独立片段。它字面上命中问题里的每个词，向量分与
     重排分都排第一，却不含任何内容 —— 是纯粹的「占位噪声」。
     评估中它正是第 12 题零命中的直接原因，因此建索引时直接剔除。

  ② 记录父子关系
     每个片段保存前一片段的完整文本（prev），命中后可向上补齐上下文。
     RAG 的经典矛盾是「小块检索准、大块上下文全」，父子分块取两者之长：
     用 200 字符的小块做精确匹配，命中后把相邻的 200 字符也一起交给
     大模型，使答案所需的连续上下文得以恢复。

运行：python rebuild_index.py
"""

import json
import os
import re
import sys
import time

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
MIN_CHUNK_CHARS = config.MIN_CHUNK_CHARS
MAX_SKELETON_CHARS = 30          # 纯标题片段的长度上限
MODEL_NAME = config.EMBEDDING_MODEL

# 加入中文标点，让切分点尽量落在句末（默认分隔符只认空格和换行，不适合中文）
SEPARATORS = config.SEPARATORS

# 空壳片段判定：短，且内部没有句读、没有换行 —— 即一个小标题
_SKELETON_RE = re.compile(r"^[^。！？；，\n]{1,%d}$" % MAX_SKELETON_CHARS)

# 「条目开头」的特征：1.  2.  三、  （一）  ①  「【专题…」等
# 在这些位置前插空行，让切分器优先在条目边界断开
_ENTRY_START_RE = re.compile(
    r"(?<!\n)(?="
    r"[0-9]{1,2}\.(?!\d)"          # 1. 2. 12.
    r"|[一二三四五六七八九十]{1,2}、"   # 一、 二、
    r"|（[一二三四五六七八九十]{1,2}）"  # （一）
    r"|\([一二三四五六七八九十]{1,2}\)"  # (一)
    r"|【(专题|专项)[一二三四五六七八九十]+"   # 【专题一
    r")"
)

# 验证用例：正确答案所在片段里必然出现的特征词，以及能把它检索出来的问法
PROBES = [
    ("光合作用的主要场所是", "光合作用的主要场所"),
    ("唐宋八大家是指哪八个人", "唐宋八大家"),
    ("恩格尔系数如何划分富裕程度", "恩格尔系数"),
    ("正当防卫的构成条件", "正当防卫"),
]


def is_skeleton(text):
    """判断是否为只有标题、没有内容的空壳片段"""
    s = text.strip()
    if len(s) < MIN_CHUNK_CHARS:
        return True
    return len(s) < MAX_SKELETON_CHARS and bool(_SKELETON_RE.match(s))


def is_continuation(text):
    """判断片段是否「从半句话开始」——即被切分点拦腰截断。

    这一步针对一个实测踩到的坑：交叉编码器孤立地给每个片段打分，
    一个从中间开始的碎片如果恰好覆盖了问题的关键词，会拿到虚高的分数，
    把真正写着答案的那一片挤下去。

    例：「唐宋八大家是指哪八个人？」
        碎片 `4.柳宗元，世称"柳河东"，与韩愈并称为"韩柳"…`  重排分 0.999
        正解 `1.唐宋八大家是指唐代韩愈、柳宗元…`            重排分 0.470

    片段应当以「条目开头」的形式起头：编号、中文序号、括号序号、
    专题标题、或模块标题。若开头是上一条的续写（如「聃）」「种"即表明…」），
    说明它是个碎片。
    """
    s = text.strip()
    if not s:
        return True

    # 表格里的续行（以 ｜ 开头）不算碎片
    if _CONTINUATION_RE.match(s):
        return True
    return False


# 「从中间开始」的特征：中文右括号、引号收尾、连接词起头等
_CONTINUATION_RE = re.compile(
    r"^[）)｜」』”]|"
    r"^[a-dA-D][）)]|"
    r"^[0-9]{1,2}[）)]|"
    r"^(即|也|与|和|及|并|而|但|则|故|因此|其中|包括|例如|如|所谓)"
)



def main():
    print("=" * 60)
    print("  重建知识库索引")
    print("=" * 60)

    with open(os.path.join(DATA, "knowledge.txt"), encoding="utf-8") as f:
        text = f.read()

    # ---------- 1) 切分 ----------
    # 预处理：在「条目开头」之前插入空行，让切分器优先在这里断开。
    #
    # 为什么需要：常识手册的条目形如
    #     【专题五 唐宋八大家
    #     1.唐宋八大家是指唐代韩愈、柳宗元…      ← 答案在这里
    #     2.古文运动是指…
    #     3.韩愈是唐宋古文运动发起者…
    # 默认分隔符只认句号、逗号，前两条会被并成一块，答案与题干的
    # 邻近关系被稀释成 0.47；正解那一片的得分反而不如中间截断的碎片。
    # 把「1.」「2.」这类条目边界提升为最高优先级的分隔符后，
    # 每个条目自成一块，答案片的得分回到 0.99 量级。
    marked = _ENTRY_START_RE.sub(r"\n\n\1", text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_text(marked)

    # ① 过滤噪声片段：
    #    · 长度不足 MIN_CHUNK_CHARS 的孤立标点
    #    · 只含一个标题、没有实质内容的空壳片段
    before = len(chunks)
    dropped = [c.strip() for c in chunks if is_skeleton(c)]
    chunks = [c for c in chunks if not is_skeleton(c)]
    if len(chunks) != before:
        print(f"\n已过滤 {before - len(chunks)} 个空壳 / 噪声片段：")
        for d in dropped:
            print(f"    · {d[:40]}")

    lengths = [len(c) for c in chunks]

    print(f"\n文档长度：{len(text)} 字符")
    print(f"切分参数：chunk_size={CHUNK_SIZE}，chunk_overlap={CHUNK_OVERLAP}")
    print(f"片段数：{len(chunks)}")
    print(f"片段长度：最短 {min(lengths)}，最长 {max(lengths)}，"
          f"平均 {sum(lengths) // len(lengths)} 字符")

    with open(os.path.join(DATA, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("\n已保存 chunks.json")

    # ② 父子关系：每个片段记录前一片段，
    #    命中后由 rag.py 向上补齐，恢复被切分点截断的上下文
    parents = [{"prev": chunks[i - 1] if i > 0 else "", "next": ""}
               for i in range(len(chunks))]
    for i in range(len(chunks) - 1):
        parents[i]["next"] = chunks[i + 1]

    with open(os.path.join(DATA, "parents.json"), "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    print("已保存 parents.json（父子关系）")


    # ---------- 2) 向量化 ----------
    print(f"\n加载模型 {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print("正在编码…")
    t0 = time.time()
    emb = model.encode(chunks, normalize_embeddings=True,
                       show_progress_bar=False)
    print(f"  完成，用时 {time.time() - t0:.1f} 秒，形状 {emb.shape}")

    # ---------- 3) 建索引 ----------
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index, os.path.join(DATA, "faiss.index"))
    print(f"\n已保存 faiss.index（{index.ntotal} 个向量）")

    # ---------- 4) 验证 ----------
    print("\n" + "=" * 60)
    print("  验证：关键常识条目的检索排名")
    print("=" * 60)

    for q, keyword in PROBES:
        q_emb = model.encode([q], normalize_embeddings=True)
        scores, ids = index.search(q_emb.astype("float32"), len(chunks))

        targets = [i for i, c in enumerate(chunks) if keyword in c]
        if not targets:
            print(f"\n  ❌ 「{keyword}」不在知识库中")
            continue

        pos = min(list(ids[0]).index(t) for t in targets)
        hit = scores[0][pos]
        mark = "✅" if pos < 5 else ("⚠️" if pos < 20 else "❌")
        print(f"\n  {mark} {q}")
        print(f"     含「{keyword}」的片段排名：第 {pos + 1}"
              f"（相似度 {hit:.4f}）")

    print("\n" + "=" * 60)
    print("  完成。若出现 ⚠️ / ❌，可下调 CHUNK_SIZE 后重跑")
    print("=" * 60)


if __name__ == "__main__":
    main()
