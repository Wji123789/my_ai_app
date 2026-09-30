# -*- coding: utf-8 -*-
"""
build_knowledge.py —— 把五份公考常识 PDF 合并为知识库 knowledge.txt

处理要点：
  1. PDF 提取的文字有大量硬换行（排版导致），按标点/编号规则重新拼行
  2. 修复上标错位：如 "3×10\\n8米/秒" -> "3×10^8米/秒"
  3. 表格单独感知提取：find_tables 找到表格区域，按行渲染成
     "单元格｜单元格｜单元格" 的形式，避免多列内容被压成一串
  4. 保留结构层级：一、> 二、> 【专题/专项】> 一、二、三 > 1. 2. 3.
  5. 修复标题空格：【专项一中共党史】 -> 【专项一 中共党史】

文档结构（合并为一个常识大模块，保留细分标题）：
    公考常识手册
    一、政治常识
    　　【专项一 中共党史】…
    二、人文常识
    　　【专题一 诸子百家】…
    …（共五个细分小类）

运行：python build_knowledge.py
输入：PDF_DIR 下的五个 PDF
输出：data/knowledge.txt
"""

import os
import re

import pymupdf

PDF_DIR = r"C:\Users\Administrator\Desktop"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge.txt")

# 合并后的唯一大类名
BOOK_TITLE = "公考常识手册"

# 顺序即细分小类「一、二、三…」的编号
PDF_PATHS = [
    (r"政治常识.pdf", "政治常识"),
    (r"人文常识.pdf", "人文常识"),
    (r"科技常识.pdf", "科技常识"),
    (r"法律常识.pdf", "法律常识"),
    (r"经济常识.pdf", "经济常识"),
]

CN_NUM = "一二三四五六"


# ------------------------------------------------------------
# 表格渲染
# ------------------------------------------------------------
def render_table(tbl) -> str:
    """把表格渲染成多行文本，每行 = 单元格用「｜」连接。

    跨行条目（如"孔子"占两行）通过"首列为空 = 续行"规则合并到上一条。
    """
    rows = tbl.extract()
    lines = []
    prev_cells = None
    for row in rows:
        cells = [re.sub(r"\s+", "", c or "") for c in row]
        cells = [c for c in cells if c]
        if not cells:
            continue
        # 续行：该行去重后是上一行的子集开头（跨页/跨行条目），直接拼接
        raw = [re.sub(r"\s+", "", c or "") for c in row]
        if prev_cells is not None and not raw[0]:
            lines[-1] += "；" + "｜".join(cells)
        else:
            lines.append("｜".join(cells))
        prev_cells = raw
    return "\n".join(f"　{line}" for line in lines)


# ------------------------------------------------------------
# 页面内容：表格区域单独处理，其余按普通文本
# ------------------------------------------------------------
def page_content(page) -> str:
    tabs = page.find_tables().tables
    tboxes = [t.bbox for t in tabs]

    def in_table(x0, y0, x1, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return any(bx0 - 2 <= cx <= bx1 + 2 and by0 - 2 <= cy <= by1 + 2
                   for (bx0, by0, bx1, by1) in tboxes)

    items = []  # (y坐标, 类型, 内容)
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0 or in_table(x0, y0, x1, y1):
            continue
        items.append((y0, "text", txt))
    for t in tabs:
        items.append((t.bbox[1], "table", render_table(t)))

    items.sort(key=lambda x: x[0])
    return "\n".join(content for _, _, content in items)


# ------------------------------------------------------------
# 断行修复与格式清理
# ------------------------------------------------------------
START_PAT = re.compile(
    r"^(第[一二三四五六七八九十]+部分|【.*|专项[一二三四五六七八九十]+"
    r"|[一二三四五六七八九十]+、|[0-9]+[.．、]|（[一二三四五六七八九十]+）)"
)


def repair_lines(raw: str) -> str:
    out, buf = [], ""
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 表格渲染行以全角空格开头（strip 会把它去掉，必须看原始行）
        is_table_line = line.startswith("　")
        if not buf or is_table_line or buf.startswith("　"):
            if buf:
                out.append(buf)
            buf = s
            continue
        prev_ends = buf.endswith(("。", "？", "！", "；", "：", "」", "）"))
        if prev_ends or START_PAT.match(s):
            out.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        out.append(buf)
    return "\n".join(out)


def fix_superscript(text: str) -> str:
    text = re.sub(r"(×10)\s*\n\s*(\d)", r"\1^\2", text)
    text = re.sub(r"(×10)\s+(?=[0-9])", r"\1^", text)
    return text


def fix_spacing(text: str) -> str:
    """【专项一中共党史】 -> 【专项一 中共党史】；一、政治常识"""
    text = re.sub(r"【(专项|专题)([一二三四五六七八九十]+)】?", r"【\1\2 ", text)
    # 上面会把「【专题三】二十四节气」变成「【专题三 二十四节气」，正合适
    text = re.sub(r"^(一、)(?=[\u4e00-\u9fff])", r"\1", text, flags=re.M)
    return text


def main():
    parts = []
    for i, (fname, label) in enumerate(PDF_PATHS, 1):
        doc = pymupdf.open(os.path.join(PDF_DIR, fname))
        raw = "\n".join(page_content(page) for page in doc)
        doc.close()

        text = fix_superscript(raw)
        text = repair_lines(text)
        text = fix_spacing(text)

        # 去掉 PDF 自带的「第X部分 XX常识」首行（与我们的标题重复）
        text = re.sub(rf"^第[一二三四五六七八九十]+部分\s*{label}\s*\n+", "", text)
        # 也去掉「XX常识之常见名词」之类的小标题
        text = re.sub(rf"^{label}之[^\n]{{0,12}}\n+", "", text)

        # 细分小类标题：一、政治常识
        header = f"{CN_NUM[i-1]}、{label}"
        parts.append(header + "\n" + text.strip())
        print(f"  {fname:<14} 整理后 {len(text):>6} 字符")

    full = f"{BOOK_TITLE}\n\n" + "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"\n已写入 {OUT}　总字符 {len(full)}")

    print("\n--- 文档结构 ---")
    for m in re.finditer(r"^([^\n]{2,20})$", full, re.M):
        pass
    for m in re.finditer(r"^[一二三四五]、\S+$", full, re.M):
        print("  " + m.group())

    print("\n--- 质检抽查 ---")
    for kw in ["【专项一 中共党史】", "×10^", "孔子", "二氧化碳", "基尼系数", "唐宋八大家"]:
        idx = full.find(kw)
        mark = "✓" if idx >= 0 else "✗ 未找到"
        print(f"  [{kw}] {mark}")
        if idx >= 0:
            print("      …" + full[max(0, idx - 30):idx + 80].replace("\n", " | ") + "…")


if __name__ == "__main__":
    main()
