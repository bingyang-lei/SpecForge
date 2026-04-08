#!/bin/bash
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
ulimit -Sn 150000
# codealpaca-20k, nemotron-math_and_code, nemotron-stem, openmath-reasoning-rl, math-stack_overflow-rl
# ================== 配置区 ==================
dataset_name="openmath-reasoning-rl"
USE_NO_THINK=false          # true=添加_no_think和--no-think参数, false=不添加
# =========================================

if [ "$USE_NO_THINK" = true ]; then
    output_suffix="_no_think_qwen3-4b_regen.jsonl"
    NO_THINK_ARG="--no-think"
    echo "当前模式: 添加 _no_think + --no-think 参数"
else
    output_suffix="_qwen3-4b_regen.jsonl"
    NO_THINK_ARG=""
    echo "当前模式: 不添加 _no_think"
fi

echo "Dataset : ${dataset_name}"
echo "Output  : ${dataset_name}${output_suffix}"

python scripts/regenerate_train_data.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003 \
    --temperature 0 \
    --input-file-path ./cache/dataset/${dataset_name}.jsonl \
    --output-file-path ./cache/dataset/${dataset_name}${output_suffix} \
    ${NO_THINK_ARG}

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
# localhost:30004 localhost:30005 localhost:30006 localhost:30007 