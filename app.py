"""
app.py —— Streamlit 主程序

按照指导书 5-2 步骤3 的分层原则：
  app.py  只负责界面与流程编排
  rag.py  负责检索与生成
  llm.py  负责模型调用（重试、异常、用量）
  prompts.py 负责提示词
  config.py 负责参数配置

界面与业务逻辑分离的好处：改界面不影响逻辑，改逻辑不用动界面。

运行（必须在终端里，不能用 IDLE 的 F5）：
    streamlit run app.py
"""

import streamlit as st

import config
import llm
import prompts
import rag

st.set_page_config(
    page_title="公考常识学习助手",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner="正在加载向量模型与知识库…")
def _load():
    """把重对象交给 Streamlit 缓存，整个会话只加载一次"""
    return rag.load_resources()


# ---------- 启动检查 ----------
try:
    model, index, chunks, mods, parents = _load()
except FileNotFoundError as e:
    st.error(f"知识库文件缺失\n\n{e}")
    st.stop()
except llm.LLMError as e:
    st.error(f"模型配置有误\n\n{e}")
    st.stop()
except Exception as e:                      # noqa: BLE001 —— 兜底，见下
    # 这里刻意捕获所有未预期的异常。
    # 原因：Streamlit 默认会把异常详情涂黑（防泄漏），云端排查时只剩一行
    # “ValueError: 本应用遇到了错误”，拿不到任何线索。
    # 与其让页面变成一片无法诊断的红色，不如把真实错误直接摊开。
    import traceback
    st.error(f"启动失败：{type(e).__name__}: {e}")
    with st.expander("查看完整堆栈（排查用）", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

MODULES = rag.modules_present()


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 检索设置")

    module = st.selectbox(
        "限定范围",
        ["全部"] + MODULES,
        help="知识库已合并为一部《公考常识手册》，其下分政治/人文/科技/法律/经济"
             "五类。明确知道考点属于哪一类时，限定范围能明显减少串题。",
    )

    top_k = st.slider("检索片段数 top_k", 1, 8, config.DEFAULT_TOP_K,
                      help="太小可能检索不全，太大引入无关内容并增加 token 开销。"
                           "默认 5 是本项目评估得出的最优值（见 evaluate.py）")

    use_rerank = st.toggle(
        "启用 Rerank 重排",
        value=config.RERANK_ENABLED,
        help="两阶段检索：先向量召回 20 条候选，再用交叉编码器精排。"
             "准确率更高，但首次使用需下载约 1GB 模型，"
             "在内存受限的环境下会自动降级为纯向量检索。",
    )

    min_score = st.slider("相似度阈值", 0.0, 0.8, config.DEFAULT_MIN_SCORE, 0.05,
                          help="低于该阈值的检索结果会被丢弃；"
                               "设为 0 表示不过滤。用于拦截知识库之外的问题。")

    use_rewrite = st.checkbox(
        "启用查询改写",
        value=False,
        help="把疑问句改写成陈述式查询再检索。实测对本知识库收益有限，"
             "且每次问答多消耗一次 API 调用，默认关闭。",
    )

    st.divider()
    st.caption("**知识库**")
    st.write(f"手册：{config.BOOK_TITLE}")
    st.write(f"文档片段：{len(chunks)}")
    st.write(f"索引向量：{index.ntotal}")
    st.write(f"覆盖分类：{len(MODULES)} 类（{'、'.join(MODULES)}）")

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
st.title("🎓 公考常识学习助手")
st.caption(
    f"基于《{config.BOOK_TITLE}》构建 —— "
    "政治 / 人文 / 科技 / 法律 / 经济 五大分类 · RAG 知识问答 + 备考计划生成"
)

tab_qa, tab_plan = st.tabs(["💬 常识问答", "📅 备考计划"])

# ============================================================
# 页签一：常识问答
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
                        st.caption(f"片段 {i}　得分 {score:.4f}")
                        st.text(text[:400] + ("…" if len(text) > 400 else ""))

    question = st.chat_input("请输入你的问题，例如：正当防卫的构成条件是什么？")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("检索并生成回答…"):
                try:
                    answer, hits = rag.answer_question(
                        question,
                        top_k=top_k,
                        min_score=min_score,
                        module="" if module == "全部" else module,
                        use_rewrite=use_rewrite,
                        use_rerank=use_rerank,
                    )
                except llm.LLMError as e:
                    answer, hits = f"⚠️ 调用失败：{e}", []

            st.markdown(answer)
            if hits:
                with st.expander(f"查看引用来源（{len(hits)} 段）"):
                    # hits 的元素是三元组 (文本, 分数, 片段下标)，
                    # 这里第三个值（下标）仅用于调试，展示时忽略
                    for i, (text, score, _) in enumerate(hits, 1):
                        st.caption(f"片段 {i}　得分 {score:.4f}")
                        st.text(text[:400] + ("…" if len(text) > 400 else ""))

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": [(t, s) for t, s, _ in hits],
        })

# ============================================================
# 页签二：备考计划
# ============================================================
with tab_plan:
    st.markdown("描述一下你的备考情况，助手会根据常识手册的内容制定复习计划。")

    with st.form("plan_form"):
        profile = st.text_area(
            "你的情况",
            value="我是应届生，行测常识部分很弱，每天能投入 1 小时，"
                  "想在 4 周内把政治和法律两大块过一遍，"
                  "科技和人文只求眼熟。",
            height=110,
        )
        submitted = st.form_submit_button("生成备考计划", type="primary")

    if submitted:
        if not profile.strip():
            st.warning("请先描述你的情况。")
        else:
            with st.spinner("正在检索资料并制定计划…"):
                try:
                    plan, hits = rag.generate_study_plan(
                        profile,
                        module="" if module == "全部" else module,
                    )
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
                    st.markdown("### ⭐ 高频考点与易错点")
                    for k in key_points:
                        st.markdown(f"- {k}")

                note = plan.get("note")
                if note and note.strip() and note.strip() != "无":
                    st.info(note)

                if hits:
                    with st.expander("查看计划依据的资料来源"):
                        for i, (text, score, _) in enumerate(hits, 1):
                            st.caption(f"片段 {i}　得分 {score:.4f}")
                            st.text(text[:300] + ("…" if len(text) > 300 else ""))

                st.caption(llm.usage_text().replace("\n", " ｜ "))
