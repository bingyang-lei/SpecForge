cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
# sharegpt, ultrachat
dataset_name="sharegpt"
python scripts/regenerate_train_data.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003 \
    --temperature 0 \
    --input-file-path ./cache/dataset/${dataset_name}_train.jsonl \
    --output-file-path ./cache/dataset/${dataset_name}_train_qwen3-4b_regen.jsonl