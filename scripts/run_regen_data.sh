cd /mnt/shared-storage-user/leihaodi/imo/SpecForge
# sharegpt, ultrachat
<<<<<<< HEAD
dataset_name="sharegpt"
=======
ulimit -Sn 155000
dataset_name="codealpaca-20k"
>>>>>>> a130d7b (temp: collect dflash local changes)
python scripts/regenerate_train_data.py \
    --model "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B" \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003 \
    --temperature 0 \
<<<<<<< HEAD
    --input-file-path ./cache/dataset/${dataset_name}_train.jsonl \
    --output-file-path ./cache/dataset/${dataset_name}_train_qwen3-4b_regen.jsonl
=======
    --input-file-path ./cache/dataset/${dataset_name}.jsonl \
    --output-file-path ./cache/dataset/${dataset_name}_qwen3-4b_regen.jsonl
>>>>>>> a130d7b (temp: collect dflash local changes)
