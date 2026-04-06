"""
MED-HALT 评测主入口 —— 推理 + 评估一体化脚本。

使用方式：
  # 1) 先在服务器启动 vLLM 服务
  bash scripts/start_vllm_server.sh qwen3-8b

  # 2) 全量推理 + 评估（所有 7 个任务）
  python run.py --api_base http://<服务器IP>:8000 --model_name Qwen3-8B

  # 3) 只跑部分任务
  python run.py --api_base http://10.0.0.1:8000 --model_name Qwen3-8B \
      --tasks FCT fake Nota

  # 4) 随机抽样 100 条做快速测试
  python run.py --api_base http://10.0.0.1:8000 --model_name Qwen3-8B \
      --tasks FCT --sample_size 100

  # 5) 仅评估（已有推理结果）
  python run.py --eval_only --output_dir outputs/Qwen3-8B --tasks FCT Nota

  # 6) 调整并发数（A6000 24G 推荐 8-16）
  python run.py --api_base http://10.0.0.1:8000 --model_name Qwen3-8B \
      --max_concurrency 12

  # 7) 若确需开启思考模式（默认关闭）
  python run.py --api_base http://10.0.0.1:8000 --model_name Llama-3.1-8B-Instruct \
      --enable_thinking
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# 将 new/ 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ALL_TASKS, load_dataset
from evaluate import evaluate_all
from inference import InferenceConfig, run_inference, save_results


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="MED-HALT 医学幻觉评测：推理 + 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 必需参数（eval_only 模式下不需要） ────────────────────
    parser.add_argument(
        "--api_base", type=str, default="http://localhost:8000",
        help="vLLM API 地址（默认: http://localhost:8000）",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen3-8B",
        help="vLLM 服务中的模型名称（--served-model-name）",
    )

    # ── 任务选择 ──────────────────────────────────────────────
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        choices=ALL_TASKS,
        help=f"要执行的任务列表（默认: 全部）。可选: {ALL_TASKS}",
    )

    # ── 数据采样 ──────────────────────────────────────────────
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="每个任务随机抽样的条数（默认: 全量）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )

    # ── 推理参数 ──────────────────────────────────────────────
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="最大生成 token 数（默认: 512）",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="生成温度（默认: 0.1，接近贪心）",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p 采样（默认: 0.95）",
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=16,
        help="客户端并发请求数（默认: 16，A6000 24G 推荐 8-16）",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="单次请求超时秒数（默认: 120）",
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="请求失败重试次数（默认: 3）",
    )

    # ── 提示词配置 ────────────────────────────────────────────
    parser.add_argument(
        "--prompt_version", type=str, default="v0",
        choices=["v0", "v1", "v2"],
        help="提示词版本（默认: v0）",
    )
    parser.add_argument(
        "--n_shots", type=int, default=2,
        help="few-shot 示例数量（默认: 2，0 为 zero-shot）",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true",
        help="开启思考模式（默认关闭，建议仅在需要链路推理时使用）",
    )

    # ── 输出配置 ──────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: outputs/<model_name>）",
    )

    # ── 评估配置 ──────────────────────────────────────────────
    parser.add_argument(
        "--eval_only", action="store_true",
        help="仅执行评估（跳过推理，需要已有推理结果）",
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="跳过评估（仅执行推理）",
    )
    return parser.parse_args()


def main() -> None:
    """主函数：根据命令行参数执行推理和/或评估流程。"""
    args = parse_args()

    # ── 确定任务列表 ──────────────────────────────────────────
    tasks = args.tasks or ALL_TASKS

    # ── 确定输出目录 ──────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    print("=" * 60)
    print("  MED-HALT 医学幻觉评测系统")
    print(f"  模型      : {args.model_name}")
    print(f"  API 地址  : {args.api_base}")
    print(f"  任务      : {tasks}")
    print(f"  抽样数    : {args.sample_size or '全量'}")
    print(f"  并发数    : {args.max_concurrency}")
    thinking_enabled = args.enable_thinking
    print(f"  思考模式  : {'开启' if thinking_enabled else '关闭'}")
    print(f"  提示词版本: {args.prompt_version} ({args.n_shots}-shot)")
    print(f"  输出目录  : {output_dir}")
    print("=" * 60)

    # ── 推理阶段 ──────────────────────────────────────────────
    if not args.eval_only:
        thinking_enabled = args.enable_thinking
        cfg = InferenceConfig(
            api_base=args.api_base,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_concurrency=args.max_concurrency,
            timeout=args.timeout,
            max_retries=args.max_retries,
            prompt_version=args.prompt_version,
            n_shots=args.n_shots,
            disable_thinking=not thinking_enabled,
        )

        for task in tasks:
            print(f"\n{'─' * 40}")
            print(f"  任务: {task}")
            print(f"{'─' * 40}")

            # 加载数据集
            dataset = load_dataset(task)
            print(f"  数据集大小: {len(dataset)} 条")

            # 随机抽样
            samples: list[dict[str, Any]] | None = None
            if args.sample_size and args.sample_size < len(dataset):
                samples = random.sample(dataset, args.sample_size)
                print(f"  抽样后: {len(samples)} 条")
            else:
                samples = dataset

            # 执行推理
            t0 = time.time()
            results = run_inference(task, cfg, samples)
            elapsed = time.time() - t0

            # 统计错误数
            errors = sum(1 for r in results if r["error"])
            print(f"  完成: {len(results)} 条, 耗时 {elapsed:.1f}s, 错误 {errors} 条")

            # 保存结果
            pred_path = output_dir / f"{task}.json"
            save_results(results, samples, pred_path)
            print(f"  已保存: {pred_path}")

    # ── 评估阶段 ──────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'=' * 60}")
        print("  评估阶段")
        print(f"{'=' * 60}")

        incorrect_scores = [0.0, -0.25]
        all_results = []
        for inc_score in incorrect_scores:
            score_label = f"incorrect={inc_score}"
            print(f"\n  ── 评分模式: {score_label} ──")
            df = evaluate_all(output_dir, tasks, 1.0, inc_score)
            if not df.empty:
                df["incorrect_score"] = inc_score
                df["model"] = args.model_name
                all_results.append(df)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            results_path = output_dir / "results.csv"
            final_df.to_csv(results_path, index=False)
            print(f"\n  评估结果已保存: {results_path}")
            print(f"\n{final_df.to_string(index=False)}")
        else:
            print("\n  未找到任何可评估的推理结果。")


if __name__ == "__main__":
    main()
