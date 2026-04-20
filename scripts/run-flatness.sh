cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
ulimit -Sn 150000
# codealpaca-20k, nemotron-math_and_code, nemotron-stem, openmath-reasoning-rl, math-stack_overflow-rl
dataset_name="nemotron-stem_qwen3-4b_regen"
output_dir_path="./cache/flatness/2"
python scripts/flatness.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --batch-size 16 \
    --max-tokens 4096 \
    --num-workers 4 \
    --temperature 0 \
    --input-file-path /mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/codealpaca-20k_qwen3-4b_regen.jsonl \
    --output-dir-path "${output_dir_path}"


# ai写的代码还是有问题，运行中的gpu占用会增加，导致爆显存
