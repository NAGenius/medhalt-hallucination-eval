"""
推理模块 —— 通过 vLLM OpenAI 兼容 API 进行异步并发推理。

核心设计：
- 使用 asyncio + aiohttp 直接调用 /v1/chat/completions 端点
- 通过信号量控制并发数，避免 24G 单卡 OOM（A6000 推荐 8-16 并发）
- 支持流式进度条显示
- 内置重试机制

关于并发安全性：
  vLLM 内部有 continuous batching 机制，会自动将并发请求合并为 GPU batch，
  因此客户端并发不会导致显存翻倍，而是由 vLLM 的 --max-num-seqs 参数控制。
  客户端信号量只是为了避免同时排队过多请求导致超时。
"""

import asyncio
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from config import (
    build_few_shot_examples,
    build_system_prompt,
    format_user_message,
    load_dataset,
)


@dataclass
class InferenceConfig:
    """推理配置参数。"""

    api_base: str = "http://localhost:8000"
    model_name: str = "Qwen3-8B"
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    seed: int = 42
    # 客户端并发数（不是 GPU batch，vLLM 内部自行调度）
    max_concurrency: int = 16
    # 单次请求超时（秒）
    timeout: int = 120
    # 失败重试次数
    max_retries: int = 3
    # 提示词版本
    prompt_version: str = "v0"
    # few-shot 数量
    n_shots: int = 2
    # 是否禁用思考模式（Qwen3 / DeepSeek-R1 等支持）
    disable_thinking: bool = True


def _strip_think_block(text: str) -> str:
    """移除模型可能返回的 <think> 思考块，仅保留最终答案区域。"""
    cleaned = text.strip()
    lower = cleaned.lower()

    close_tag = "</think>"
    if close_tag in lower:
        end_idx = lower.rfind(close_tag)
        return cleaned[end_idx + len(close_tag):].strip()

    # 兜底：只有开标签且输出被截断时，尽量从第一个左大括号开始恢复。
    if lower.startswith("<think>"):
        first_brace = cleaned.find("{")
        if first_brace != -1:
            return cleaned[first_brace:].strip()

    return cleaned


def _extract_brace_candidates(text: str) -> list[str]:
    """提取文本中所有平衡的大括号片段，便于从混合文本恢复 JSON。"""
    candidates: list[str] = []
    start_idx: int | None = None
    depth = 0

    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(text[start_idx:idx + 1])
                start_idx = None

    # 优先尝试较长片段（通常信息更完整）
    candidates.sort(key=len, reverse=True)
    return candidates


def _try_parse_json(text: str) -> dict[str, Any]:
    """尝试从模型输出中解析 JSON。

    支持多种常见格式：纯 JSON、markdown 代码块、混合文本中的 JSON。

    Args:
        text: 模型原始输出文本。

    Returns:
        解析后的字典，解析失败返回空字典。
    """
    text = text.strip()
    if not text:
        return {}

    text = _strip_think_block(text)

    # 去除 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # 尝试直接解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 从文本中提取平衡的大括号片段，逐个尝试解析。
    for candidate in _extract_brace_candidates(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    return {}


async def _call_chat_completion(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    cfg: InferenceConfig,
    system_prompt: str,
    user_message: str,
    sample_id: str,
) -> dict[str, Any]:
    """发送单次 chat completion 请求。

    Args:
        session: aiohttp 会话。
        semaphore: 并发控制信号量。
        cfg: 推理配置。
        system_prompt: 系统提示词。
        user_message: 用户消息。
        sample_id: 样本 ID，用于结果追踪。

    Returns:
        包含 id、raw_output、parsed_output、error 的字典。
    """
    url = f"{cfg.api_base}/v1/chat/completions"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    payload: dict[str, Any] = {
        "model": cfg.model_name,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed": cfg.seed,
        "stop": ["Stop Here"],
    }

    # 对思考模型在请求体顶层显式关闭 thinking。
    if cfg.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(1, cfg.max_retries + 1):
        async with semaphore:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=cfg.timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"API 返回 {resp.status}: {error_text[:300]}")

                    result = await resp.json()
                    raw_output = result["choices"][0]["message"]["content"]
                    return {
                        "id": sample_id,
                        "raw_output": raw_output,
                        "parsed_output": _try_parse_json(raw_output),
                        "error": None,
                    }
            except Exception as e:
                if attempt == cfg.max_retries:
                    return {
                        "id": sample_id,
                        "raw_output": "",
                        "parsed_output": {},
                        "error": f"{type(e).__name__}: {e}",
                    }
                await asyncio.sleep(1.0 * attempt)

    # 理论上不会到这里
    return {"id": sample_id, "raw_output": "", "parsed_output": {}, "error": "未知错误"}


async def run_inference_async(
    task: str,
    cfg: InferenceConfig,
    samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """对指定任务的数据集执行异步并发推理。

    Args:
        task: 任务短名。
        cfg: 推理配置。
        samples: 可选的数据子集，为 None 时加载完整数据集。

    Returns:
        推理结果列表，每条包含 id、raw_output、parsed_output、error。
    """
    if samples is None:
        samples = load_dataset(task)

    system_prompt = build_system_prompt(task, cfg.prompt_version)
    few_shot_text = build_few_shot_examples(task, cfg.n_shots)

    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sample in samples:
            user_msg = format_user_message(task, sample, few_shot_text)
            coro = _call_chat_completion(
                session,
                semaphore,
                cfg,
                system_prompt,
                user_msg,
                sample["id"],
            )
            tasks.append(coro)

        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"  推理 [{task}]",
            total=len(tasks),
        )

    return results


def run_inference(
    task: str,
    cfg: InferenceConfig,
    samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """同步入口：对指定任务执行推理（内部使用 asyncio）。

    Args:
        task: 任务短名。
        cfg: 推理配置。
        samples: 可选的数据子集。

    Returns:
        推理结果列表。
    """
    return asyncio.run(run_inference_async(task, cfg, samples))


def save_results(
    results: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """将推理结果与原始数据合并后保存为 JSON。

    保存格式与原始论文代码的 eval 输入兼容。

    Args:
        results: 推理结果列表。
        dataset: 原始数据集列表。
        output_path: 输出文件路径。
    """
    id_to_result = {result["id"]: result for result in results}

    merged = []
    for sample in dataset:
        sample_id = sample["id"]
        result = id_to_result.get(sample_id, {})
        merged.append(
            {
                "id": sample_id,
                "testbed_data": sample,
                "gpt_output": result.get("parsed_output", {}),
                "raw_output": result.get("raw_output", ""),
                "error": result.get("error"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
