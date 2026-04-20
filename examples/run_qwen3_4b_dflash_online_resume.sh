#!/bin/bash
# 先等待1800s

cd /mnt/shared-storage-user/leihaodi/imo/SpecForge/
# pip install -e .
# python -m pip install /mnt/shared-storage-user/leihaodi/imo/flashinfer_pkgs/flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  num_gpu=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  num_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

ATTENTION_BACKEND=${2:-flex_attention}
USE_KL=false
KL_ALPHA=0.1
IGNORE_GRAD_NORM=false
USE_EAGLE3_LOSS=false
KL_ARGS=""
WANDB_SUFFIX=""
GRAD_NORM_ARGS=""
EAGLE3_LOSS_ARGS=""

if [ "$USE_KL" = true ]; then
  KL_ARGS="--use-kl --kl-alpha $KL_ALPHA"
  WANDB_SUFFIX="_kl${KL_ALPHA}"
fi
if [ "$IGNORE_GRAD_NORM" = true ]; then
  GRAD_NORM_ARGS="--ignore-grad-norm"
fi
if [ "$USE_EAGLE3_LOSS" = true ]; then
  EAGLE3_LOSS_ARGS="--use-eagle3-loss"
fi
echo "KL_ARGS: $KL_ARGS"
echo "EAGLE3_LOSS_ARGS: $EAGLE3_LOSS_ARGS"

sleep 3

# TRAIN_DATA_PATH='[
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-math_and_code_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-stem_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/new-code-dataset/code_merged_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/openmath-reasoning-rl_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/math-stack_overflow-rl_qwen3-4b_regen.jsonl"
# ]'

RESUME_CKPT="$ROOT_DIR/outputs/qwen3-4b-dflash-oss/epoch_1_step_80000"
if [ ! -d "$RESUME_CKPT" ]; then
  echo "Resume checkpoint not found: $RESUME_CKPT"
  exit 1
fi
echo "Resuming training from checkpoint: $RESUME_CKPT"
TRAIN_DATA_PATH='[
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-math_and_code_qwen3-4b_regen.jsonl",
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k_qwen3-4b_regen.jsonl",
    "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/nemotron-stem_qwen3-4b_regen.jsonl"
]'

# TRAIN_DATA_PATH='[
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/acp_app_bool_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/acp_app_gen_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/aime24_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/aime25_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/gsm8k_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/math500_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/mbpp_qwen3-4b_regen.jsonl",
#     "/mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/test_dataset/humaneval_qwen3-4b_regen.jsonl"
# ]'

torchrun \
    --standalone \
    --nproc_per_node $num_gpu \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflash-oss.json \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-dflash-oss \
    --resume \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 3e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 2048 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 768 \
    --loss-decay-gamma 7.0 \
    --log-interval 500 \
    --save-interval 10000 \
    --report-to wandb \
    --wandb-offline \
    --wandb-project specforge-qwen3-4b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    $KL_ARGS \
    $GRAD_NORM_ARGS \
    $EAGLE3_LOSS_ARGS

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py

# bash /mnt/shared-storage-user/leihaodi/imo/SpecForge/examples/run_qwen3_4b_dflash_online.sh flex_attention
# --report-to wandb # none
# 32,4 : 71000   320,4 : 104000  320,2 : 87000  320,1 : 76000   32,1 : 70000  
# 1600,1 : 118000   1600,1,4096 : 128000（增加的没有那么快）

