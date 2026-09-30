"""
evaluate.py —— 评估与迭代（实验5-3）

思路：用「大模型当裁判（LLM-as-Judge）」建立自动评估流程，
      摆脱纯人工试用的主观性。

用法：
    python evaluate.py run v1              # 跑一轮评估，结果存为 eval_v1.json
    python evaluate.py run v6 --rerank     # 开启 Rerank 重排跑一轮
    python evaluate.py compare v5 v6       # 对比两轮结果
    python evaluate.py show v1             # 查看某轮明细

测试用例覆盖三类（指导书要求）：
    · 核心功能 —— 典型使用场景
    · 边界情况 —— 空输入、超长输入、无关问题
    · 对抗情况 —— 试图诱导模型脱离知识库

测试用例存放在 testsets/gongkao.json，与代码分离 ——
换一套题目不用改代码，也方便对同一套题做多轮对比。
"""

import argparse
import json
import os
import sys

import config
import llm
import prompts
import rag

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTSET_DIR = os.path.join(BASE_DIR, "testsets")
DEFAULT_TESTSET = "gongkao"

# 检索片段数。实测最优值，与 app.py 默认值一致。
TOP_K = config.DEFAULT_TOP_K


def load_testset(name=DEFAULT_TESTSET):
    """从 testsets/<name>.json 读取测试用例"""
    path = os.path.join(TESTSET_DIR, name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到测试集：{path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data["cases"]


def judge_one(question, reference, answer):
    """用大模型给单条回答打分（temperature=0，保证评分稳定）"""
    msgs = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user",
         "content": prompts.judge(question=question,
                                  reference=reference, answer=answer)},
    ]
    result = llm.chat_json(msgs, temperature=0.0, max_tokens=400)
    if result is None:
        return {"accuracy": 0, "completeness": 0, "relevance": 0,
                "reason": "评分失败"}
    return result


def run(label, top_k=TOP_K, use_rerank=None, use_rewrite=False,
        module="", testset=DEFAULT_TESTSET):
    """跑一轮完整评估"""
    cases = load_testset(testset)
    rerank_state = "开启" if (use_rerank or
                              (use_rerank is None and config.RERANK_ENABLED)) \
        else "关闭"

    print("=" * 64)
    print(f"  评估运行：{label}")
    print("=" * 64)
    print(f"  测试集：{testset}（{len(cases)} 条）")
    print(f"  配置：top_k={top_k}　rerank={rerank_state}　"
          f"rewrite={'开' if use_rewrite else '关'}　"
          f"module={module or '全部'}")
    print(f"  每条 2 次调用（回答 + 评分），共约 {len(cases) * 2} 次\n")

    rag.load_resources()
    if use_rerank or (use_rerank is None and config.RERANK_ENABLED):
        rag.load_reranker()          # 预热，避免把加载时间算进单题耗时
    llm.reset_usage()

    records = []
    for case in cases:
        print(f"  [{case['id']:>2}/{len(cases)}] {case['q'][:38]}…")

        try:
            answer, hits = rag.answer_question(
                case["q"], top_k=top_k, module=module,
                use_rewrite=use_rewrite, use_rerank=use_rerank)
        except llm.LLMError as e:
            answer, hits = f"（调用失败：{e}）", []

        score = judge_one(case["q"], case["ref"], answer)
        top = hits[0][1] if hits else 0.0

        records.append({
            "id": case["id"],
            "type": case.get("type", ""),
            "q": case["q"],
            "ref": case["ref"],
            "answer": answer,
            "top_score": round(top, 4),
            "n_hits": len(hits),
            **{k: score.get(k, 0) for k in
               ("accuracy", "completeness", "relevance", "reason")},
        })
        print(f"        准确{records[-1]['accuracy']} "
              f"完整{records[-1]['completeness']} "
              f"相关{records[-1]['relevance']}  得分 {top:.3f}")

    # ---------- 汇总 ----------
    n = len(records)
    avg = {k: sum(r[k] for r in records) / n
           for k in ("accuracy", "completeness", "relevance")}
    avg["total"] = sum(avg.values()) / 3

    print("\n" + "=" * 64)
    print(f"  汇总（{label}）")
    print("=" * 64)
    print(f"  准确性均分：{avg['accuracy']:.2f}")
    print(f"  完整性均分：{avg['completeness']:.2f}")
    print(f"  相关性均分：{avg['relevance']:.2f}")
    print(f"  ── 综合均分：{avg['total']:.2f} / 5.00")

    # 按类型分组统计（定位低分的共同特征）
    print("\n  按用例类型：")
    for t in ("核心功能", "边界情况", "对抗情况"):
        sub = [r for r in records if r["type"] == t]
        if sub:
            s = sum((r["accuracy"] + r["completeness"] + r["relevance"]) / 3
                    for r in sub) / len(sub)
            print(f"    {t}（{len(sub)} 条）：{s:.2f}")

    # 低分用例
    low = sorted(records, key=lambda r: r["accuracy"] + r["completeness"]
                 + r["relevance"])[:3]
    print("\n  得分最低的 3 条：")
    for r in low:
        s = (r["accuracy"] + r["completeness"] + r["relevance"]) / 3
        print(f"    [{r['id']}] {s:.2f} 分　{r['q'][:32]}…")
        print(f"         裁判：{r['reason'][:60]}")

    print(f"\n  {llm.usage_text().replace(chr(10), ' ｜ ')}")

    out = os.path.join(BASE_DIR, f"eval_{label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": label, "testset": testset,
                   "config": {"top_k": top_k, "rerank": rerank_state,
                              "rewrite": use_rewrite, "module": module},
                   "avg": avg, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存：eval_{label}.json")

    # 同时维护一份固定的「最终结果」副本，供报告与文档引用
    final_path = os.path.join(BASE_DIR, "eval_final.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump({"label": label, "testset": testset,
                   "config": {"top_k": top_k, "rerank": rerank_state,
                              "rewrite": use_rewrite, "module": module},
                   "avg": avg, "records": records},
                  f, ensure_ascii=False, indent=2)
    return avg


def compare(label_a, label_b):
    """对比两轮评估结果"""
    pa = os.path.join(BASE_DIR, f"eval_{label_a}.json")
    pb = os.path.join(BASE_DIR, f"eval_{label_b}.json")
    for p in (pa, pb):
        if not os.path.exists(p):
            print(f"❌ 找不到 {p}")
            return

    with open(pa, encoding="utf-8") as f:
        A = json.load(f)
    with open(pb, encoding="utf-8") as f:
        B = json.load(f)

    print("=" * 64)
    print(f"  改进前后对比：{label_a}  →  {label_b}")
    print("=" * 64)
    print(f"\n  {'维度':<10}{label_a:>12}{label_b:>12}{'变化':>12}")
    print("  " + "-" * 44)
    for k, name in (("accuracy", "准确性"), ("completeness", "完整性"),
                    ("relevance", "相关性"), ("total", "综合")):
        d = B["avg"][k] - A["avg"][k]
        arrow = "↑" if d > 0.01 else ("↓" if d < -0.01 else "—")
        print(f"  {name:<10}{A['avg'][k]:>12.2f}{B['avg'][k]:>12.2f}"
              f"{arrow + ' ' + format(abs(d), '.2f'):>12}")

    # 逐题对比
    print(f"\n  {'#':<4}{label_a:>8}{label_b:>8}{'变化':>8}   问题")
    print("  " + "-" * 60)
    for ra, rb in zip(A["records"], B["records"]):
        sa = (ra["accuracy"] + ra["completeness"] + ra["relevance"]) / 3
        sb = (rb["accuracy"] + rb["completeness"] + rb["relevance"]) / 3
        d = sb - sa
        mark = f"+{d:.2f}" if d > 0.01 else (f"{d:.2f}" if d < -0.01 else "—")
        print(f"  {ra['id']:<4}{sa:>8.2f}{sb:>8.2f}{mark:>8}   {ra['q'][:28]}…")


def show(label):
    """查看某轮评估的明细"""
    path = os.path.join(BASE_DIR, f"eval_{label}.json")
    if not os.path.exists(path):
        print(f"❌ 找不到 {path}")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"=== {label} 明细（综合均分 {data['avg']['total']:.2f}）===\n")
    for r in data["records"]:
        s = (r["accuracy"] + r["completeness"] + r["relevance"]) / 3
        print(f"[{r['id']}] {r['type']}　{s:.2f} 分　得分 {r['top_score']}")
        print(f"  问：{r['q'][:60]}")
        print(f"  答：{r['answer'][:150]}…")
        print(f"  评：{r['reason'][:70]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG 应用评估")
    ap.add_argument("cmd", choices=["run", "compare", "show"])
    ap.add_argument("labels", nargs="*", help="run: 轮次名；compare: 两个轮次名")
    ap.add_argument("--rerank", action="store_true", help="开启 Rerank 重排")
    ap.add_argument("--no-rerank", action="store_true", help="关闭 Rerank 重排")
    ap.add_argument("--rewrite", action="store_true", help="开启查询改写")
    ap.add_argument("--module", default="", help="限定模块")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--testset", default=DEFAULT_TESTSET)
    args = ap.parse_args()

    if args.cmd == "run":
        label = args.labels[0] if args.labels else "v1"
        use_rerank = None
        if args.rerank:
            use_rerank = True
        elif args.no_rerank:
            use_rerank = False
        run(label, top_k=args.top_k, use_rerank=use_rerank,
            use_rewrite=args.rewrite, module=args.module,
            testset=args.testset)
    elif args.cmd == "compare":
        if len(args.labels) < 2:
            print("用法：python evaluate.py compare v1 v2")
        else:
            compare(args.labels[0], args.labels[1])
    else:
        show(args.labels[0] if args.labels else "v1")
