"""
vLLM RESTful API 连通性验证脚本

自动检测当前加载的模型，根据模型类型选择合适的接口（Chat / Completions）。

验证内容：
  1. 服务健康检查（/health）
  2. 模型列表查询（/v1/models）
  3. 文本生成验证（自动适配 Chat 或 Completions 接口）

使用方式：
  python scripts/test_vllm_api.py
  python scripts/test_vllm_api.py --api_base http://192.168.1.100:8000
"""

import argparse
import sys
import time
from typing import Any

import requests

TIMEOUT = 120  # 秒

# 不支持 Chat 接口的基础补全模型（没有 chat template）
BASE_MODELS = {"Llama-2-7B"}


# ── 工具函数 ───────────────────────────────────────────────
def print_sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print(f"\n{'─' * width}")


def check_pass(label: str) -> None:
    print(f"  [PASS] {label}")


def check_fail(label: str, reason: str) -> None:
    print(f"  [FAIL] {label}: {reason}")


# ── 各项验证 ───────────────────────────────────────────────
def test_health(api_base: str) -> bool:
    """验证服务健康状态"""
    print_sep("1. 健康检查")
    try:
        resp = requests.get(f"{api_base}/health", timeout=TIMEOUT)
        if resp.status_code == 200:
            check_pass(f"GET /health → {resp.status_code}")
            return True
        check_fail("GET /health", f"状态码 {resp.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        check_fail("GET /health", "无法连接，请确认服务已启动")
        return False


def test_models(api_base: str) -> str | None:
    """查询已加载模型，返回模型名称"""
    print_sep("2. 模型列表")
    resp = requests.get(f"{api_base}/v1/models", timeout=TIMEOUT)
    data = resp.json()
    model_ids = [m["id"] for m in data.get("data", [])]
    print(f"  已加载模型: {model_ids}")
    if model_ids:
        model_name = model_ids[0]
        check_pass(f"检测到模型: '{model_name}'")
        return model_name
    check_fail("模型列表", "无已加载模型")
    return None


def test_generation(api_base: str, model_name: str) -> bool:
    """
    根据模型类型自动选择接口进行生成测试。
    - 基础模型 → /v1/completions
    - 指令/对话模型 → /v1/chat/completions
    """
    is_base = model_name in BASE_MODELS
    endpoint = "completions" if is_base else "chat/completions"
    print_sep(f"3. 生成测试（/{endpoint}）")

    # 构造请求
    prompt_text = (
        "Question: What is the most common cause of community-acquired pneumonia?\n"
        "A. Klebsiella pneumoniae\n"
        "B. Streptococcus pneumoniae\n"
        "C. Staphylococcus aureus\n"
        "D. Haemophilus influenzae\n"
        "Answer:"
    )

    params: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }

    if is_base:
        # Completions 接口：纯文本补全
        params["prompt"] = prompt_text
    else:
        # Chat 接口
        params["messages"] = [{"role": "user", "content": prompt_text}]
        # 思考模型（Qwen3 / DeepSeek-R1）禁用 thinking
        if any(k in model_name.lower() for k in ("qwen3", "deepseek-r1")):
            params["chat_template_kwargs"] = {"enable_thinking": False}

    url = f"{api_base}/v1/{endpoint}"
    print(f"  请求地址: {url}")

    t0 = time.time()
    try:
        resp = requests.post(url, json=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        check_fail("请求失败", f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        return False
    elapsed = time.time() - t0

    data = resp.json()

    # 提取生成内容
    if is_base:
        content = data["choices"][0]["text"]
    else:
        content = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})
    print(f"  响应内容: {repr(content[:300])}")
    print(f"  耗时    : {elapsed:.2f}s")
    print(f"  Token   : prompt={usage.get('prompt_tokens')}  "
          f"completion={usage.get('completion_tokens')}")

    if content and content.strip():
        check_pass("模型生成正常")
        return True
    check_fail("生成验证", "返回内容为空")
    return False


# ── 主流程 ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM API 验证脚本")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000",
                        help="vLLM 服务地址")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    print(f"\nvLLM API 验证脚本")
    print(f"目标服务: {api_base}")

    # 1. 健康检查
    if not test_health(api_base):
        sys.exit(1)

    # 2. 模型列表
    model_name = test_models(api_base)
    if not model_name:
        sys.exit(1)

    # 3. 生成测试
    gen_ok = test_generation(api_base, model_name)

    # 汇总
    print_sep("验证汇总")
    results = {"健康检查": True, "模型列表": True, "文本生成": gen_ok}
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(results.values())
    total = len(results)
    print(f"\n  共 {total} 项，通过 {passed} 项，失败 {total - passed} 项")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
