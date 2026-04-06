"""
任务配置模块 —— 定义七个 MED-HALT 评测任务的提示词、数据加载和格式化逻辑。

所有任务统一使用 JSON 数据集（datasets/ 目录），通过 vLLM OpenAI 兼容 API 进行推理。
"""

import json
import random
import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# ── 路径常量 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROMPTS_DIR = PROJECT_ROOT / "medhalt" / "prompts"

# ── 任务注册表 ────────────────────────────────────────────────

# 任务短名 → 数据集文件名（不含扩展名）
TASK_DATASET_MAP: dict[str, str] = {
    "FCT": "reasoning_FCT",
    "fake": "reasoning_fake",
    "Nota": "reasoning_nota",
    "pmid2title": "IR_pmid2title",
    "url2title": "IR_pubmedlink2title",
    "title2pub": "IR_title2pubmedlink",
    "abs2pub": "IR_abstract2pubmedlink",
}

# 任务短名 → prompts 子目录名
TASK_PROMPT_DIR: dict[str, str] = {
    "FCT": "reasoning_FCT",
    "fake": "reasoning_Fake",
    "Nota": "Reasoning_Nota",
    "pmid2title": "IR_pmid2title",
    "url2title": "IR_pubmedlink2title",
    "title2pub": "IR_title2pubmedlink",
    "abs2pub": "IR_abstract2pubmedlink",
}

# 任务分组
REASONING_TASKS = {"FCT", "fake", "Nota"}
IR_TASKS = {"pmid2title", "url2title", "title2pub", "abs2pub"}
ALL_TASKS = list(TASK_DATASET_MAP.keys())


# ── 数据加载 ──────────────────────────────────────────────────

def load_dataset(task: str) -> list[dict[str, Any]]:
    """加载指定任务的 JSON 数据集。

    Args:
        task: 任务短名，如 'FCT'、'pmid2title' 等。

    Returns:
        数据集记录列表。

    Raises:
        FileNotFoundError: 数据集文件不存在。
        KeyError: 未知任务名。
    """
    dataset_name = TASK_DATASET_MAP[task]
    path = DATASETS_DIR / f"{dataset_name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 提示词构建 ────────────────────────────────────────────────

def _load_prompts_json(task: str) -> dict:
    """加载任务对应的 prompts.json。"""
    path = PROMPTS_DIR / TASK_PROMPT_DIR[task] / "prompts.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_shots_json(task: str) -> list[dict]:
    """加载任务对应的 shots.json，返回示例列表。"""
    path = PROMPTS_DIR / TASK_PROMPT_DIR[task] / "shots.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["shots"][0]


def build_system_prompt(task: str, version: str = "v0") -> str:
    """构建系统提示词（system message）。

    Args:
        task: 任务短名。
        version: 提示词版本，'v0'、'v1' 或 'v2'。

    Returns:
        系统提示词字符串。
    """
    prompts_data = _load_prompts_json(task)
    prompt_obj = next(p for p in prompts_data["prompts"] if p["id"] == version)
    return prompt_obj["prompt"] + "\n" + prompt_obj["output_format"]


def build_few_shot_examples(task: str, n_shots: int = 2) -> str:
    """构建 few-shot 示例文本。

    Args:
        task: 任务短名。
        n_shots: 示例数量，0 表示 zero-shot。

    Returns:
        格式化后的示例文本，zero-shot 时返回空字符串。
    """
    if n_shots == 0:
        return ""

    shots = _load_shots_json(task)
    default_shots = [s for s in shots if s["prompt_type"] == "default"]
    task_shots = [s for s in shots if s["prompt_type"] != "default"]

    # 根据 n_shots 数量分配 default 和 task-specific 示例
    if n_shots == 1:
        selected = random.sample(default_shots, min(1, len(default_shots)))
    elif n_shots == 2:
        selected = random.sample(default_shots, min(1, len(default_shots)))
        selected += random.sample(task_shots, min(1, len(task_shots)))
    elif n_shots <= 5:
        n_default = min(n_shots - n_shots // 2, len(default_shots))
        n_task = min(n_shots // 2, len(task_shots))
        selected = random.sample(default_shots, n_default)
        selected += random.sample(task_shots, n_task)
    else:
        selected = random.sample(shots, min(n_shots, len(shots)))

    lines = ["Examples:"]
    for s in selected:
        lines.append(f"Input : {s['input']}")
        lines.append(f"Output: {s['Output']}Stop Here")
        lines.append("")
    return "\n".join(lines)


# ── 单条数据 → 用户消息 ──────────────────────────────────────

def format_user_message(task: str, sample: dict[str, Any],
                        few_shot_text: str = "") -> str:
    """将一条数据集记录格式化为用户消息（user message）。

    Args:
        task: 任务短名。
        sample: 数据集中的一条记录。
        few_shot_text: few-shot 示例文本（可选）。

    Returns:
        拼接好的用户消息字符串。
    """
    if task in REASONING_TASKS:
        # Reasoning 任务：question + options
        options = sample["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        body = f"Input: {{'Question': {sample['question']!r}, 'Options': {options}}}\nOutput: "
    elif task == "pmid2title":
        body = f"Input: {{'Pmid': '{int(sample['PMID'])}'}}\nOutput: "
    elif task == "abs2pub":
        body = f"Input: {{'paper_abstract': {sample['Abstract']!r}}}\nOutput: "
    elif task == "url2title":
        body = f"Input: {{'url': {sample['url']!r}}}\nOutput: "
    elif task == "title2pub":
        body = f"Input: {{'paper_title': {sample['Title']!r}}}\nOutput: "
    else:
        raise ValueError(f"未知任务: {task}")

    if few_shot_text:
        return few_shot_text + "\n" + body
    return body
