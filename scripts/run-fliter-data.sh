cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
ulimit -Sn 150000
# # codealpaca-20k, nemotron-math_and_code, nemotron-stem, openmath-reasoning-rl, math-stack_overflow-rl
# # ================== 配置区 ==================
# dataset_name="math-stack_overflow-rl"

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

python scripts/nll-fliter-data.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address "${server_addresses[@]}" \
    --temperature 0 \
    --input-file-path /mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/dataset/math-stack_overflow-rl_qwen3-4b_regen.jsonl \
    --output-file-path /mnt/shared-storage-user/leihaodi/imo/SpecForge/cache/nll-fliter-data/math-stack_overflow-rl_qwen3-4b_regen_nll_top50.jsonl

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py

