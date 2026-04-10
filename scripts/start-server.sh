#!/usr/bin/env bash
# Launch one SGLang server per GPU on ports 30000, 30001, ...
# GPU count is read from nvidia-smi. Only the last server's logs go to the terminal.
# Override: CUDA_VISIBLE_DEVICES=0,1 ./start-server.sh (still uses visible GPU count from nvidia-smi)

set -euo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
MODEL="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3.5-4B"

if ! command -v nvidia-smi &>/dev/null; then
  echo "Error: nvidia-smi not found; cannot detect GPU count." >&2
  exit 1
fi

gpu_num=$(nvidia-smi -L 2>/dev/null | wc -l)
gpu_num=$(echo "${gpu_num}" | tr -d '[:space:]')

if [[ -z "${gpu_num}" || "${gpu_num}" -lt 1 ]]; then
  echo "Error: no GPUs reported by nvidia-smi (gpu_num=${gpu_num:-0})." >&2
  exit 1
fi

last_gpu=$((gpu_num - 1))

for i in $(seq 0 "${last_gpu}"); do
  port=$((30000 + i))
  echo "Starting SGLang on port ${port} (CUDA_VISIBLE_DEVICES=${i})..."
  if [[ "${i}" -eq "${last_gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${i}" python3 -m sglang.launch_server \
      --model "${MODEL}" \
      --cuda-graph-bs 1 2 4 8 16 32 64 128 \
      --dtype bfloat16 \
      --mem-frac=0.8 \
      --attention-backend fa3 \
      --mamba-scheduler-strategy extra_buffer \
      --port "${port}" &
  else
    CUDA_VISIBLE_DEVICES="${i}" python3 -m sglang.launch_server \
      --model "${MODEL}" \
      --cuda-graph-bs 1 2 4 8 16 32 64 128 \
      --dtype bfloat16 \
      --mem-frac=0.8 \
      --attention-backend fa3 \
      --mamba-scheduler-strategy extra_buffer \
      --port "${port}" >/dev/null 2>&1 &
  fi
done

echo "All ${gpu_num} server(s) started in background. Check PIDs with: jobs -l"
wait
