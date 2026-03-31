cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
ulimit -Sn 150000
# odealpaca-20k, nemotron-math_and_code, nemotron-stem
dataset_name="nemotron-stem"
python scripts/regenerate_train_data.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003 localhost:30004 localhost:30005 localhost:30006 localhost:30007 \
    --temperature 0 \
    --no-think \
    --input-file-path ./cache/dataset/${dataset_name}.jsonl \
    --output-file-path ./cache/dataset/${dataset_name}_no_think_qwen3-4b_regen.jsonl

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
