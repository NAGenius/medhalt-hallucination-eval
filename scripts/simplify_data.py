"""
将 MED-HALT 数据集 CSV 文件转换为 JSON 格式，保留每个任务所需的完整字段。

对 IR_pmid2title / IR_pubmedlink2title 中的 fake_data 样本，
将 Title（原值 Unknown）修正为 source_title（真实标题），
并保留 original_title 字段记录修正前的原始值，用于论文复现。

各任务保留字段：
  IR_pmid2title           : id, PMID, Title, original_title
  IR_pubmedlink2title     : id, url, Title, original_title
  IR_title2pubmedlink     : id, Title, url
  IR_abstract2pubmedlink  : id, Abstract, url
    reasoning_FCT           : id, question, options, cop, cop_index
  reasoning_fake          : id, question, options
    reasoning_nota          : id, question, options, cop, cop_index
"""

import json
import pandas as pd
from pathlib import Path
import math

INPUT_DIR = Path("./medhalt/datasets")
OUTPUT_DIR = Path("./datasets")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 需要修正 fake_data Title 的任务
FIXUP_TASKS = {"IR_pmid2title", "IR_pubmedlink2title"}

# 各任务需要保留的列（修正任务额外输出 original_title）
TASK_COLUMNS: dict[str, list[str]] = {
    "IR_pmid2title":          ["id", "PMID", "Title"],
    "IR_pubmedlink2title":    ["id", "url", "Title"],
    "IR_title2pubmedlink":    ["id", "Title", "url"],
    "IR_abstract2pubmedlink": ["id", "Abstract", "url"],
    "reasoning_FCT":          ["id", "question", "options", "correct_answer", "correct_index"],
    "reasoning_fake":         ["id", "question", "options"],
    "reasoning_nota":         ["id", "question", "options", "correct_answer", "correct_index"],
}


def save_json(records: list[dict], name: str) -> None:
    """将记录列表写入 JSON 文件。

    Args:
        records: 待写入的记录列表。
        name:    不含扩展名的文件名，输出至 OUTPUT_DIR/{name}.json。
    """
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        safe_records = _sanitize_for_json(records)
        json.dump(safe_records, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[OK] {name}.json  ({len(records)} 条)")


def _sanitize_for_json(value):
    """递归清洗对象，确保 JSON 中不会出现 NaN。"""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


for task_name, columns in TASK_COLUMNS.items():
    csv_path = INPUT_DIR / f"{task_name}.csv"

    if task_name in FIXUP_TASKS:
        # 需要额外读取 source_title 和 pubmed_data_type 来修正 Title
        read_cols = columns + ["source_title", "pubmed_data_type"]
        # keep_default_na=False: 避免把字符串 "None" 自动当成 NaN
        df = pd.read_csv(csv_path, usecols=read_cols, keep_default_na=False)

        # 保留原始 Title 值
        df["original_title"] = df["Title"]

        # fake_data 行：用 source_title 替换 Unknown
        mask = df["pubmed_data_type"] == "fake_data"
        df.loc[mask, "Title"] = df.loc[mask, "source_title"]

        out_cols = columns + ["original_title"]
        df = df.astype(object).where(df.notna(), other=None)
        records: list[dict] = df[out_cols].to_dict(orient="records")  # type: ignore[assignment]
    else:
        # keep_default_na=False: 避免把字符串 "None" 自动当成 NaN
        df = pd.read_csv(csv_path, usecols=columns, keep_default_na=False)

        # 与原始论文提示词字段对齐：correct_answer -> cop, correct_index -> cop_index
        if task_name in {"reasoning_FCT", "reasoning_nota"}:
            df = df.rename(columns={
                "correct_answer": "cop",
                "correct_index": "cop_index",
            })
            columns = ["id", "question", "options", "cop", "cop_index"]

        df = df.astype(object).where(df.notna(), other=None)
        records = df[columns].to_dict(orient="records")  # type: ignore[assignment]

    save_json(records, task_name)

print(f"\nDone. 输出目录：{OUTPUT_DIR}")
