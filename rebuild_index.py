"""
rebuild_index.py —— 重建知识库索引

背景：
    实验四使用 chunk_size=300 切分，得到 98 个片段。
    实验五的评估发现「注意事项」类内容（如 robots 协议、请求频率）
    被埋在长片段里，向量被稀释，检索排名靠后，取不到。

    对比实验（目标片段「robots」的检索排名）：
        chunk_size=300 -> 第 21 名
        chunk_size=200 -> 第 4 名    ← 本项目采用
        chunk_size=150 -> 第 3 名
        chunk_size=120 -> 第 3 名

    更细的分块提升了检索精度，但也会削弱上下文完整性，
    因此不宜过细 —— 实测 chunk_size=150 时多项用例反而退化。
    折中取 200。

运行：python rebuild_index.py
"""

import json
import os
import time

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 加入中文标点，让切分点尽量落在句末（默认分隔符只认空格和换行，不适合中文）
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def main():
    print("=" * 60)
    print("  重建知识库索引")
    print("=" * 60)

    with open(os.path.join(DATA, "knowledge.txt"), encoding="utf-8") as f:
        text = f.read()

    # ---------- 1) 切分 ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    lengths = [len(c) for c in chunks]

    print(f"\n文档长度：{len(text)} 字符")
    print(f"切分参数：chunk_size={CHUNK_SIZE}，chunk_overlap={CHUNK_OVERLAP}")
    print(f"片段数：{len(chunks)}")
    print(f"片段长度：最短 {min(lengths)}，最长 {max(lengths)}，"
          f"平均 {sum(lengths) // len(lengths)} 字符")

    with open(os.path.join(DATA, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("\n已保存 chunks.json")

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
    print("  验证：robots 内容的检索排名")
    print("=" * 60)

    q = "实验三中，采集网络数据需要注意哪些问题？"
    q_emb = model.encode([q], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype("float32"), len(chunks))

    targets = [i for i, c in enumerate(chunks) if "robots" in c.lower()]
    if targets:
        pos = min(list(ids[0]).index(t) for t in targets)
        print(f"  问题：{q}")
        print(f"  含 robots 的片段排名：第 {pos + 1}（相似度 {scores[0][pos]:.4f}）")
        print(f"\n  命中片段内容：")
        print("  " + chunks[targets[0]][:200].replace("\n", "\n  "))
    else:
        print("  ❌ 知识库中找不到 robots 相关内容")


if __name__ == "__main__":
    main()
