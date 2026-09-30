"""
config.py —— 全局配置中心

所有可调参数集中在这里，其他模块只从这里读，不自己定义。
好处：换模型、改价格、调检索参数只需要改这一个文件，
也不用担心同一参数在几个文件里改了这处漏了那处。

环境变量优先级高于这里的默认值（部署平台不改代码也能调整）。
"""

import os

# ---------- 大模型 API ----------
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

# 单价（元 / 百万 token）
LLM_INPUT_PRICE = float(os.environ.get("LLM_INPUT_PRICE", "1.0"))
LLM_OUTPUT_PRICE = float(os.environ.get("LLM_OUTPUT_PRICE", "4.0"))

# 重试策略：指数退避 1s -> 2s -> 4s
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_BASE_WAIT = float(os.environ.get("LLM_BASE_WAIT", "1.0"))

# ---------- 向量与重排模型 ----------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")

# 重排开关：本地默认开；内存受限的部署环境可设 RERANK_ENABLED=0 关闭。
# 即使开着，加载失败（如云端内存不足）也会自动降级为普通检索，不会让应用崩掉。
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") == "1"
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "20"))  # 参与重排的候选数

# ---------- 检索参数 ----------
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))
DEFAULT_MIN_SCORE = float(os.environ.get("DEFAULT_MIN_SCORE", "0.0"))

# 汇总类问题的扩大召回：
# 「这个模块都包含哪些内容」这类问题需要跨章节汇总，答案分散在几十个片段里，
# 常规 top-k 召回必然不全。检测到此意图时把候选池放大，提高覆盖概率。
# 注意：扩大召回只能提高「碰得到」的概率，不能让答案从池外进来 ——
# 这是 FAISS top-k 的结构性限制，已在实验五报告中作为结论说明。
AGGREGATION_POOL_MULTIPLIER = int(os.environ.get("AGG_POOL", "2"))

# 父子分块：命中片段后向上补齐相邻片段，恢复被切分点截断的上下文
PARENT_CHILD_ENABLED = os.environ.get("PARENT_CHILD", "1") == "1"
PARENT_MAX_EXPAND = int(os.environ.get("PARENT_MAX_EXPAND", "1"))  # 向上补齐几片
PARENT_CHAR_BUDGET = int(os.environ.get("PARENT_CHAR_BUDGET", "420"))  # 单片段上下文上限

# 知识库的细分小类（对应 knowledge.txt 中「一、政治常识」这类小节标题）。
# 五份手册已合并为一个「公考常识」大模块，这里是它下面的细分小类，
# 仍可按小类限定检索范围。
MODULES = ["政治", "人文", "科技", "法律", "经济"]

# 合并后的知识库大类名
BOOK_TITLE = "公考常识手册"

# ---------- 知识库切分（rebuild_index.py 使用） ----------
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "40"))
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "10"))  # 短于此的噪声片段丢弃
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
