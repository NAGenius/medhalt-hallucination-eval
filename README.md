# 医学大语言模型幻觉评测系统（毕业论文）

本项目基于 MED-HALT 数据集，评测医学大模型在推理（Reasoning）和信息检索（Information Retrieval）任务上的幻觉表现。

## 项目目标

- 统一管理七个评测任务的数据、提示词、推理与评估流程。
- 支持通过 vLLM OpenAI 兼容接口进行批量推理。
- 产出可复现实验结果，服务毕业论文实验部分。

## 七个评测任务

- Reasoning
  - FCT
  - fake
  - Nota
- IR
  - pmid2title
  - url2title
  - title2pub
  - abs2pub

## 目录结构（核心）

- new/
  - config.py：任务与数据、提示词配置
  - inference.py：并发推理（vLLM API）
  - evaluate.py：任务评估逻辑
  - run.py：推理+评估一体化入口
- datasets/
  - 七个任务的 JSON 数据
- medhalt/
  - 原始代码与提示词资源
- scripts/
  - simplify_data.py：CSV -> JSON 数据预处理
  - test_vllm_api.py：vLLM 接口检查

## 环境建议

- Python 3.10+
- 建议使用虚拟环境

## 快速开始

1. 生成 JSON 数据集（如尚未生成）

   python scripts/simplify_data.py

2. 启动 vLLM 服务（示例）

   bash scripts/start_vllm_server.sh qwen3-8b

3. 运行全部任务推理与评估

   python new/run.py --api_base http://localhost:8000 --model_name Qwen3-8B

4. 仅运行部分任务

   python new/run.py --api_base http://localhost:8000 --model_name Qwen3-8B --tasks FCT fake Nota

5. 若确需开启思考模式

   python new/run.py --api_base http://localhost:8000 --model_name Qwen3-8B --enable_thinking

## 输出说明

- 推理结果默认输出到 outputs/<model_name>/
- 评估汇总写入 outputs/<model_name>/results.csv

## 复现建议

- 固定随机种子（默认 seed=42）
- 记录模型版本、提示词版本和评估参数
- 将关键实验配置写入提交信息，便于追溯
