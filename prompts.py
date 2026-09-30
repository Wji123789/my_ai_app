"""
prompts.py —— 提示词模板管理

提示词全部存放在 prompts/ 目录下的 .md 文件里，与代码分离（指导书 5-1 步骤2）。

这样做的价值：
  · 迭代提示词不需要改 Python 代码，也不用担心改错逻辑
  · 提示词可以单独做版本管理，方便 A/B 对比不同版本的效果
  · 非程序员也能参与提示词调优

目录结构：
  prompts/
    system_prompt.md          # 全局角色设定
    task_prompts/
      rag_qa.md               # RAG 问答
      study_plan.md           # 学习计划生成
      judge.md                # 评估打分
"""

import os

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_cached = {}


def load(name):
    """读取提示词模板。

    name 取值：
      "system_prompt"           -> prompts/system_prompt.md
      "task_prompts/rag_qa"     -> prompts/task_prompts/rag_qa.md
    """
    if name in _cached:
        return _cached[name]

    path = os.path.join(PROMPT_DIR, name + ".md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到提示词模板：{path}")

    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    _cached[name] = text
    return text


def render(name, **kwargs):
    """读取模板并填充变量。

    模板里用 {变量名} 占位；若模板中需要字面量花括号（比如 JSON 示例），
    写成双花括号 {{ }}。
    """
    return load(name).format(**kwargs)


# ---------- 便捷入口 ----------
def system_prompt():
    """全局角色设定"""
    return load("system_prompt")


def rag_qa(context, question):
    """RAG 问答提示词"""
    return render("task_prompts/rag_qa", context=context, question=question)


def study_plan(profile, context):
    """学习计划生成提示词"""
    return render("task_prompts/study_plan", profile=profile, context=context)


def judge(question, reference, answer):
    """评估打分提示词"""
    return render("task_prompts/judge",
                  question=question, reference=reference, answer=answer)


def list_prompts():
    """列出所有提示词模板，便于检查"""
    result = []
    for root, _, files in os.walk(PROMPT_DIR):
        for f in files:
            if f.endswith(".md"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, PROMPT_DIR).replace("\\", "/")
                result.append(rel[:-3])
    return sorted(result)


if __name__ == "__main__":
    print("提示词模板库：")
    for p in list_prompts():
        text = load(p)
        print(f"  {p:<32} {len(text):>4} 字符")
