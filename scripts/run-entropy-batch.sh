#!/bin/bash
set -euo pipefail

cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
ulimit -Sn 150000

MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3.5-4B"
FIG_COUNT=10000
FIG_ROOT="/mnt/shared-storage-user/leihaodi/imo/SpecForge/qwen3.5-entropy_results"
mkdir -p "${FIG_ROOT}"
DATASETS=(
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-stem-40960lines.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/openmath-reasoning-rl.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/ultrachat_train.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/sharegpt_train.jsonl"
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-math-first10wlines.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-code-last5wlines.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/math-stack_overflow-rl.jsonl"
    # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k.jsonl"
)

num_gpu=$(nvidia-smi -L | wc -l)
if [ "$num_gpu" -le 0 ]; then
    echo "未检测到 GPU，无法生成 server-address"
    exit 1
fi

server_addresses=()
for ((i=0; i<num_gpu; i++)); do
    server_addresses+=("localhost:$((30000 + i))")
done

echo "检测到 GPU 数量: ${num_gpu}"
echo "使用 server addresses: ${server_addresses[*]}"
echo "图像输出根目录: ${FIG_ROOT}"
echo

for input_file_path in "${DATASETS[@]}"; do
    dataset_file=$(basename "${input_file_path}")
    dataset_name="${dataset_file%.jsonl}"

    for think_mode in on off; do
        fig_dir="${FIG_ROOT}/${dataset_name}-${think_mode}"
        mkdir -p "${fig_dir}"

        echo "========================================"
        echo "Dataset   : ${dataset_name}"
        echo "Think mode: ${think_mode}"
        echo "Input     : ${input_file_path}"
        echo "Output dir: ${fig_dir}"
        echo "========================================"

        cmd=(
            python scripts/regenerate_train_data.py
            --model "${MODEL_PATH}"
            --concurrency 128
            --max-tokens 3072
            --server-address "${server_addresses[@]}"
            --temperature 0
            --input-file-path "${input_file_path}"
            --entropy
            --sample-num "${FIG_COUNT}"
            --fig-dir "${fig_dir}"
            --ignore-fig
        )

        if [ "${think_mode}" = "off" ]; then
            cmd+=(--no-think)
        fi

        "${cmd[@]}"
        echo
    done
done

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
