"""测试：不同的 chunk_size 能否让「注意事项」类内容被检索到"""
import json
import os

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

with open(os.path.join(DATA, "knowledge.txt"), encoding="utf-8") as f:
    text = f.read()

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

SEPS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

Q = "实验三中，采集网络数据需要注意哪些问题？"
KEY = "robots"

q_emb = model.encode([Q], normalize_embeddings=True)

print(f"测试问题：{Q}")
print(f"期望命中关键词：{KEY}\n")
print(f"{'chunk_size':>12}{'overlap':>10}{'片段数':>10}{'含关键词片段':>14}{'该片段排名':>12}")
print("-" * 62)

for size, overlap in [(300, 50), (200, 40), (150, 30), (120, 25)]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=SEPS, length_function=len,
    )
    chunks = sp.split_text(text)

    emb = model.encode(chunks, normalize_embeddings=True,
                       show_progress_bar=False)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype(np.float32))

    scores, ids = idx.search(q_emb.astype(np.float32), len(chunks))

    targets = [i for i, c in enumerate(chunks) if KEY in c.lower()]
    if targets:
        pos = min(list(ids[0]).index(t) for t in targets)
        rank = f"第 {pos + 1}"
        if pos < 3:
            rank += " ✅"
    else:
        rank = "未找到"

    print(f"{size:>12}{overlap:>10}{len(chunks):>10}"
          f"{len(targets):>14}{rank:>12}")
