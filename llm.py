"""
llm.py —— 大模型调用封装层

职责：所有与大模型 API 的交互都必须经过这里。
      业务代码（app.py）不直接 import openai，只管调用 chat()/chat_json()。

为什么要单独抽这一层（指导书 5-2 步骤1）：
  1. 统一重试：网络抖动、限流(429)、服务端错误(5xx) 都能自动恢复
  2. 统一异常：调用方只需捕获 LLMError，不用认识 openai 的各种异常类型
  3. 集中统计：token 用量与成本在一个地方算，不用到处埋点
  4. 便于替换：将来换平台、换 SDK、加代理，只改这个文件
"""

import json
import logging
import os
import time

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

import config

logger = logging.getLogger("llm")

# ---------- 配置（集中在 config.py，本文件不再自定义） ----------
MODEL = config.LLM_MODEL
BASE_URL = config.LLM_BASE_URL
INPUT_PRICE = config.LLM_INPUT_PRICE
OUTPUT_PRICE = config.LLM_OUTPUT_PRICE
MAX_RETRIES = config.LLM_MAX_RETRIES
BASE_WAIT = config.LLM_BASE_WAIT

# 可重试的异常：这些通常都是临时故障，过一会儿重试就能成功
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class LLMError(Exception):
    """本项目统一的模型调用异常。

    调用方只需要捕获这一个类型，不必关心底层用的是哪个 SDK、
    也不用区分是网络问题还是配额问题——错误信息里已经写清楚了。
    """


# ---------- 密钥管理 ----------
def _get_api_key():
    """获取密钥，支持两种来源。

    本地运行  -> 读环境变量 DEEPSEEK_API_KEY
    云端部署  -> 读部署平台的 Secrets（Streamlit Community Cloud 用
                 .streamlit/secrets.toml 管理，界面上叫 Secrets）

    两种来源都不写死在代码里，密钥也不会进版本库
    （.gitignore 已排除 .env 与 secrets.toml）。
    """
    # 1) 环境变量（本地开发）
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key

    # 2) Streamlit Secrets（云端部署）
    #    没有安装 streamlit 或不在 Streamlit 环境里时，这里会抛异常，忽略即可
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return None


def _build_client():
    """构造客户端。密钥绝不写进代码，也不进版本库。"""
    api_key = _get_api_key()
    if not api_key:
        raise LLMError(
            "未找到大模型密钥。请按运行环境任选一种方式配置：\n\n"
            "【本地运行】设置环境变量后重开终端：\n"
            '  [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY","sk-xxx","User")\n\n'
            "【云端部署】在部署平台的 Secrets 中配置：\n"
            '  DEEPSEEK_API_KEY = "sk-xxx"'
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


_client = None


def get_client():
    """惰性初始化客户端（避免 import 时就因缺少密钥而失败）"""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------- 用量统计 ----------
USAGE = {"calls": 0, "prompt": 0, "completion": 0, "retries": 0}


def reset_usage():
    USAGE.update({"calls": 0, "prompt": 0, "completion": 0, "retries": 0})


def cost():
    return (USAGE["prompt"] / 1e6 * INPUT_PRICE
            + USAGE["completion"] / 1e6 * OUTPUT_PRICE)


def usage_text():
    total = USAGE["prompt"] + USAGE["completion"]
    return (f"调用 {USAGE['calls']} 次"
            + (f"（含重试 {USAGE['retries']} 次）" if USAGE["retries"] else "")
            + f"\n输入 {USAGE['prompt']} + 输出 {USAGE['completion']} "
              f"= {total} token\n约 {cost():.6f} 元")


def _record(usage):
    USAGE["calls"] += 1
    USAGE["prompt"] += usage.prompt_tokens
    USAGE["completion"] += usage.completion_tokens


# ---------- 核心调用 ----------
def chat(messages, temperature=0.3, max_tokens=1000, json_mode=False,
         max_retries=MAX_RETRIES):
    """调用大模型，带指数退避重试与统一异常处理。

    参数
      messages    : [{"role": "...", "content": "..."}, ...]
      temperature : 采样温度。判定类用 0.0，生成类用 0.7~1.0
      json_mode   : 是否强制 JSON 输出（提示词里必须出现 "json" 字样）
    返回
      模型回答的字符串
    异常
      LLMError —— 重试耗尽后仍失败，或遇到不可重试的错误
    """
    client = get_client()

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            _record(resp.usage)
            return resp.choices[0].message.content

        except RETRYABLE as e:
            # 临时故障：退避后重试
            last_error = e
            if attempt == max_retries:
                break
            wait = BASE_WAIT * (2 ** attempt)      # 1s -> 2s -> 4s
            USAGE["retries"] += 1
            logger.warning("调用失败（%s），%.0fs 后重试（%d/%d）…",
                           type(e).__name__, wait, attempt + 1, max_retries)
            time.sleep(wait)

        except APIError as e:
            # 其他 API 错误：401 认证失败、402 余额不足等，重试没有意义
            raise LLMError(f"API 调用失败：{e}") from e

        except Exception as e:
            raise LLMError(f"未预期的错误：{type(e).__name__}: {e}") from e

    raise LLMError(
        f"连续 {max_retries + 1} 次调用均失败，最后一次错误："
        f"{type(last_error).__name__}: {last_error}"
    )


def chat_json(messages, temperature=0.0, max_tokens=1000):
    """调用大模型并解析 JSON 结果。

    模型偶尔会返回不合法的 JSON，这里做重试；
    全部失败时返回 None，由调用方决定如何降级——而不是直接崩溃。
    """
    for attempt in range(2):
        raw = chat(messages, temperature=temperature,
                   max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("第 %d 次返回的不是合法 JSON，重试…", attempt + 1)

    logger.error("JSON 解析连续失败")
    return None
