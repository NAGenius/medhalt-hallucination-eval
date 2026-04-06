#!/bin/bash
# vLLM RESTful API 启动脚本（多模型支持）
#
# 使用方式:
#   bash start_vllm_server.sh <模型名>
#
# 可用模型:
#   qwen3-8b          Qwen/Qwen3-8B（思考模型，默认禁用 thinking）
#   deepseek-r1       deepseek-ai/DeepSeek-R1-0528-Qwen3-8B（思考模型）
#   llama2-7b         LLM-Research/llama-2-7b
#   llama3.1-8b       LLM-Research/Meta-Llama-3.1-8B-Instruct
#
# 示例:
#   bash start_vllm_server.sh qwen3-8b
#   bash start_vllm_server.sh llama3.1-8b

set -e

# ── 公共配置 ──────────────────────────────────────────────
MODEL_ROOT="/mnt/helong7/毕业设计/models"
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.92

# ── 模型专属配置 ──────────────────────────────────────────
# 24G 单卡环境下，各模型的最优参数
configure_model() {
    case "$1" in

        qwen3-8b)
            MODEL_PATH="${MODEL_ROOT}/Qwen/Qwen3-8B"
            SERVED_MODEL_NAME="Qwen3-8B"
            DTYPE="bfloat16"
            # Qwen3-8B 默认 context 32K，24G 显存下需限制
            MAX_MODEL_LEN=8192
            MAX_NUM_SEQS=32
            # 无额外参数
            EXTRA_ARGS=""
            ;;

        deepseek-r1)
            MODEL_PATH="${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            SERVED_MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
            DTYPE="bfloat16"
            # DeepSeek-R1 基于 Qwen3 架构，配置类似
            MAX_MODEL_LEN=8192
            MAX_NUM_SEQS=32
            EXTRA_ARGS=""
            ;;

        llama2-7b)
            MODEL_PATH="${MODEL_ROOT}/LLM-Research/llama-2-7b"
            SERVED_MODEL_NAME="Llama-2-7B"
            DTYPE="float16"
            # LLaMA-2 原生 context 4K，基础补全模型
            MAX_MODEL_LEN=4096
            MAX_NUM_SEQS=64
            EXTRA_ARGS=""
            ;;

        llama3.1-8b)
            MODEL_PATH="${MODEL_ROOT}/LLM-Research/Meta-Llama-3.1-8B-Instruct"
            SERVED_MODEL_NAME="Llama-3.1-8B-Instruct"
            DTYPE="bfloat16"
            # LLaMA-3.1 原生 128K，24G 下需要大幅限制
            MAX_MODEL_LEN=8192
            MAX_NUM_SEQS=32
            EXTRA_ARGS=""
            ;;

        *)
            echo "错误: 未知模型 '$1'"
            echo ""
            echo "可用模型: qwen3-8b | deepseek-r1 | llama2-7b | llama3.1-8b"
            exit 1
            ;;
    esac
}

# ── 参数检查 ──────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "用法: bash $0 <模型名>"
    echo ""
    echo "可用模型:"
    echo "  qwen3-8b          Qwen/Qwen3-8B"
    echo "  deepseek-r1       deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    echo "  llama2-7b         LLM-Research/llama-2-7b"
    echo "  llama3.1-8b       LLM-Research/Meta-Llama-3.1-8B-Instruct"
    exit 1
fi

configure_model "$1"

# ── 启动信息 ──────────────────────────────────────────────
echo "========================================"
echo "  启动 vLLM API 服务"
echo "  模型名称  : $1"
echo "  模型路径  : $MODEL_PATH"
echo "  监听地址  : http://$HOST:$PORT"
echo "  别名      : $SERVED_MODEL_NAME"
echo "  数据类型  : $DTYPE"
echo "  上下文长度: $MAX_MODEL_LEN"
echo "  最大并发  : $MAX_NUM_SEQS"
echo "  显存利用率: $GPU_MEMORY_UTILIZATION"
echo "========================================"

# ── 启动服务 ──────────────────────────────────────────────
vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    $EXTRA_ARGS
