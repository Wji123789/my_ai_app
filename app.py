"""
app.py —— Streamlit 主程序

按照指导书 5-2 步骤3 的分层原则：
  app.py  只负责界面与流程编排
  rag.py  负责检索与生成
  llm.py  负责模型调用（重试、异常、用量）
  prompts.py 负责提示词

界面与业务逻辑分离的好处：改界面不影响逻辑，改逻辑不用动界面。

运行（必须在终端里，不能用 IDLE 的 F5）：
    streamlit run app.py
"""

import streamlit as st

import llm
import prompts
import rag

st.set_page_config(
    page_title="课程学习助手",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="正在加载向量模型与知识库…")
def _load():
    """把重对象交给 Streamlit 缓存，整个会话只加载一次"""
    return rag.load_resources()


# ---------- 启动检查 ----------
try:
    model, index, chunks = _load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except llm.LLMError as e:
    st.error(str(e))
    st.stop()


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 设置")

    top_k = st.slider("检索片段数 top_k", 1, 8, 5,
                      help="太小可能检索不全，太大引入无关内容并增加 token 开销。"
                           "默认 5 是本项目评估得出的最优值（见 evaluate.py）")

    min_score = st.slider("相似度阈值", 0.0, 0.8, 0.0, 0.05,
                          help="低于该阈值的检索结果会被丢弃；"
                               "设为 0 表示不过滤。用于拦截知识库之外的问题。")

    st.divider()
    st.caption("**知识库**")
    st.write(f"文档片段：{len(chunks)}")
    st.write(f"索引向量：{index.ntotal}")

    st.divider()
    st.caption("**本次会话消耗**")
    st.text(llm.usage_text())

    st.divider()
    if st.button("清空对话历史", use_container_width=True):
        st.session_state.messages = []
        llm.reset_usage()
        st.rerun()

    with st.expander("查看提示词模板"):
        for name in prompts.list_prompts():
            st.caption(f"`{name}.md`")


# ---------- 主界面 ----------
st.title("📚 课程学习助手")
st.caption("基于《大模型应用实训》实验指导书构建 —— RAG 知识问答 + 学习计划生成")

tab_qa, tab_plan = st.tabs(["💬 知识问答", "📅 学习计划"])

# ============================================================
# 页签一：知识问答
# ============================================================
with tab_qa:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"查看引用来源（{len(m['sources'])} 段）"):
                    for i, (text, score) in enumerate(m["sources"], 1):
                        st.caption(f"片段 {i}　相似度 {score:.4f}")
                        st.text(text[:400] + ("…" if len(text) > 400 else ""))

    question = st.chat_input("请输入你的问题，例如：实验四的思考题有哪些？")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("检索并生成回答…"):
                try:
                    answer, hits = rag.answer_question(
                        question, top_k=top_k, min_score=min_score)
                except llm.LLMError as e:
                    answer, hits = f"⚠️ 调用失败：{e}", []

            st.markdown(answer)
            if hits:
                with st.expander(f"查看引用来源（{len(hits)} 段）"):
                    for i, (text, score) in enumerate(hits, 1):
                        st.caption(f"片段 {i}　相似度 {score:.4f}")
                        st.text(text[:400] + ("…" if len(text) > 400 else ""))

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": hits,
        })

# ============================================================
# 页签二：学习计划
# ============================================================
with tab_plan:
    st.markdown("描述一下你的情况，助手会根据课程内容制定学习计划。")

    with st.form("plan_form"):
        profile = st.text_area(
            "你的情况",
            value="我是编程初学者，每天能投入 2 小时，想在 4 周内完成这门课的五个实验，"
                  "重点想搞懂 RAG 那部分。",
            height=110,
        )
        submitted = st.form_submit_button("生成学习计划", type="primary")

    if submitted:
        if not profile.strip():
            st.warning("请先描述你的情况。")
        else:
            with st.spinner("正在检索资料并制定计划…"):
                try:
                    plan, hits = rag.generate_study_plan(profile)
                except llm.LLMError as e:
                    st.error(f"调用失败：{e}")
                    plan, hits = None, []

            if plan is None:
                st.error("计划生成失败（模型未返回合法 JSON），请重试。")
            else:
                st.success(f"目标：{plan.get('goal', '')}")

                stages = plan.get("stages", [])
                if stages:
                    cols = st.columns(min(len(stages), 4))
                    for i, s in enumerate(stages):
                        with cols[i % len(cols)]:
                            st.markdown(f"**{s.get('week', '')}**")
                            st.caption(s.get("focus", ""))

                for s in stages:
                    with st.expander(f"{s.get('week', '')}　{s.get('focus', '')}"):
                        st.markdown("**任务**")
                        for t in s.get("tasks", []):
                            st.markdown(f"- {t}")
                        st.markdown(f"**对应资料**：{s.get('material', '—')}")
                        st.markdown(f"**完成标准**：{s.get('outcome', '—')}")

                key_points = plan.get("key_points", [])
                if key_points:
                    st.markdown("### ⭐ 重点与易错点")
                    for k in key_points:
                        st.markdown(f"- {k}")

                note = plan.get("note")
                if note and note.strip() and note.strip() != "无":
                    st.info(note)

                if hits:
                    with st.expander("查看计划依据的资料来源"):
                        for i, (text, score) in enumerate(hits, 1):
                            st.caption(f"片段 {i}　相似度 {score:.4f}")
                            st.text(text[:300] + ("…" if len(text) > 300 else ""))

                st.caption(llm.usage_text().replace("\n", " ｜ "))
