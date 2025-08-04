#!/bin/bash
set -euo pipefail

model_type=$1
if [ "$model_type" == "qwen_7b" ]; then
    export CUDA_VISIBLE_DEVICES=0
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    output_file="./vlm_logs/vllm_7b.log"
elif [ "$model_type" == "qwen_72b" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"
    output_file="./vlm_logs/vllm_72b.log"
else
    echo -e "\033[31m[ERROR] Invalid model type: $model_type\033[0m"
    exit 1
fi

export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false # Avoid tokenizer parallelism

# 服务配置
API_PORT=8001
LOCAL_MODEL_DIR="${MODEL_NAME}"

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
echo -e "\033[34m[INFO] Using ${num_gpus} GPUs\033[0m"

nohup python -u -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 \
        --port ${API_PORT} \
        --gpu-memory-utilization 0.8 \
        --max-model-len 14336 \
        --model "${LOCAL_MODEL_DIR}" \
        --tensor-parallel-size ${num_gpus} \
        --limit-mm-per-prompt image=25 \
        --trust-remote-code \
        > ${output_file} 2>&1 &

echo -e "\033[34m[INFO] VLLM server started on port ${API_PORT}\033[0m"