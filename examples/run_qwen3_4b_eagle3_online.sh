#!/bin/bash
bash /mnt/shared-storage-user/leihaodi/imo/SpecForge/scripts/install.sh
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}  # tp到底有有啥用？调整它不改变train的时候的进度条长度啊，而且越大似乎内存占用越高了（有点离谱，理论上不是应该更低）
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# TRAIN_DATA_PATH='[
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-math_and_code_no_think_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k_no_think_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-stem_no_think_qwen3-4b_regen.jsonl"
# ]'

TRAIN_DATA_PATH='[
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-math_and_code_qwen3-4b_regen.jsonl",
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k_qwen3-4b_regen.jsonl",
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-stem_qwen3-4b_regen.jsonl"
]'

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3.json \
    --train-data-path "$TRAIN_DATA_PATH" \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-eagle3-dflash_data-think \
    --num-epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-length 3072 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --log-interval 500 \
    --save-interval 20000 \
    --target-model-backend sglang \
    --report-to wandb \
    --wandb-offline \
    --wandb-project specforge-qwen3-4b-dflash

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
