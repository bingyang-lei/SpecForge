#!/bin/bash
cd /mnt/shared-storage-user/leihaodi/imo/SpecForge/
# pip install -e .
# python -m pip install /mnt/shared-storage-user/leihaodi/imo/flashinfer_pkgs/flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-8}

ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflash.json \
    --train-data-path '["/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/ultrachat_train_qwen3-4b_regen.jsonl","/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/sharegpt_train_qwen3-4b_regen.jsonl"]' \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-ultrachat_sharegpt \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 500 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-offline \
    --wandb-project specforge-qwen3-4b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --wandb-name qwen3-4b-dflash
