"""
Compute per-sample flatness scores for existing chat datasets.

This script reads a jsonl dataset whose items contain a `conversations` field,
extracts the user / assistant contents, tokenizes the full chat with the target
model's chat template, and computes the flatness of each assistant token.

The sample flatness is the mean flatness over all scored assistant tokens.

Why this script does local forward instead of querying the running sglang
servers:
The public OpenAI-compatible sglang endpoints expose prompt logprobs, but not
the full prompt-side vocabulary distribution required by the flatness metric.
This script uses local model forward and supports multi-worker parallelism.
"""

import argparse
import json
import math
import os
import statistics
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute sample flatness scores for an existing chat dataset"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per worker (per GPU/CPU worker)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of local workers (typically number of GPUs to use)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument(
        "--output-dir-path",
        type=str,
        default=None,
        help="Directory to store output jsonl/error/summary files",
    )
    return parser.parse_args()


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def extract_messages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Input item does not contain a valid `conversations` list")

    messages = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": extract_text(message.get("content"))})

    if not messages:
        raise ValueError("No valid system/user/assistant messages found")
    return messages


def build_assistant_mask_fallback(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> List[int]:
    assistant_mask: List[int] = []
    prev_len = 0
    for end_idx in range(1, len(messages) + 1):
        tokenized = tokenizer.apply_chat_template(
            list(messages[:end_idx]),
            tokenize=True,
            add_generation_prompt=False,
        )
        curr_len = len(tokenized)
        delta_len = curr_len - prev_len
        if delta_len < 0:
            raise ValueError("Chat template token length unexpectedly decreased")
        fill_value = 1 if messages[end_idx - 1]["role"] == "assistant" else 0
        assistant_mask.extend([fill_value] * delta_len)
        prev_len = curr_len
    return assistant_mask


def tokenize_with_assistant_mask(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    has_assistant_message = any(msg.get("role") == "assistant" for msg in messages)

    try:
        tokenized = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
        )
        input_ids = tokenized["input_ids"][0]
        assistant_mask = tokenized.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = tokenized.get("assistant_tokens_mask")
        if assistant_mask is None:
            raise KeyError("assistant token mask was not returned")
        assistant_mask = assistant_mask[0].to(dtype=torch.bool)
        # Some tokenizer/template combinations return an all-zero assistant mask
        # even when assistant turns exist. In that case, fallback to robust
        # incremental reconstruction.
        if has_assistant_message and int(assistant_mask.sum().item()) == 0:
            raise ValueError("assistant token mask is empty; fallback reconstruction")
        return input_ids, assistant_mask
    except Exception:
        input_ids = torch.tensor(
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=False,
            ),
            dtype=torch.long,
        )
        assistant_mask = torch.tensor(
            build_assistant_mask_fallback(tokenizer, messages),
            dtype=torch.bool,
        )
        if assistant_mask.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"assistant mask length mismatch: {assistant_mask.shape[0]} vs {input_ids.shape[0]}"
            )
        return input_ids, assistant_mask


def flatness_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_empty((0,), dtype=torch.float32)

    effective_temperature = temperature if temperature > 0 else 1.0
    scaled_logits = logits.float() / effective_temperature
    log_z = torch.logsumexp(scaled_logits, dim=-1)
    log_z2 = torch.logsumexp(2.0 * scaled_logits, dim=-1)
    log_p_norm = 0.5 * (log_z2 - 2.0 * log_z)
    log_vocab = math.log(logits.shape[-1])
    # Cosine similarity between p_t and U=(1/V,...,1/V):
    # flatness = 1 / (sqrt(V) * ||p_t||_2)
    log_flatness = -0.5 * log_vocab - log_p_norm
    return torch.exp(log_flatness)


def default_output_path(input_file_path: str, output_dir_path: Optional[str] = None) -> str:
    input_path = Path(input_file_path)
    output_dir = Path(output_dir_path) if output_dir_path else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix == ".jsonl":
        return str(output_dir / f"{input_path.stem}_flatness.jsonl")
    return str(output_dir / f"{input_path.name}_flatness.jsonl")


def build_half_comparison(flatness_values: Sequence[float]) -> Dict[str, Any]:
    if not flatness_values:
        return {
            "num_valid_samples": 0,
            "message": "No valid flatness values to compare.",
        }

    sorted_values = sorted(float(x) for x in flatness_values)
    n = len(sorted_values)
    split_idx = n // 2
    lower_half = sorted_values[:split_idx]
    upper_half = sorted_values[split_idx:]

    def summarize(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    lower_stats = summarize(lower_half)
    upper_stats = summarize(upper_half)

    mean_lower = lower_stats.get("mean")
    mean_upper = upper_stats.get("mean")
    diff = None
    ratio = None
    if mean_lower is not None and mean_upper is not None:
        diff = float(mean_upper - mean_lower)
        ratio = float(mean_upper / mean_lower) if mean_lower != 0 else None

    return {
        "num_valid_samples": n,
        "split_rule": "sorted ascending by flatness, first floor(N/2) as lower half, remaining as upper half",
        "lower_50_percent": lower_stats,
        "upper_50_percent": upper_stats,
        "comparison": {
            "mean_diff_upper_minus_lower": diff,
            "mean_ratio_upper_div_lower": ratio,
        },
    }


@dataclass
class WorkerResult:
    status: str
    data: Dict[str, Any]


@dataclass
class PreparedSample:
    original_data: Dict[str, Any]
    input_ids: Optional[torch.Tensor]
    assistant_mask: Optional[torch.Tensor]
    total_prompt_tokens: int
    prompt_tokens_used: int
    error: Optional[str] = None


class FlatnessWorker:
    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            device_map={"": str(self.device)},
        )
        self.model.eval()

        model_body = getattr(self.model, self.model.base_model_prefix, None)
        if model_body is None:
            model_body = getattr(self.model, "model", None)
        if model_body is None:
            raise ValueError("Unable to locate the base transformer module")
        self.model_body = model_body

        if not hasattr(self.model, "lm_head"):
            raise ValueError("Model does not expose `lm_head`, cannot compute logits")
        self.lm_head = self.model.lm_head
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def _prepare_sample(self, data: Dict[str, Any], max_tokens: int) -> PreparedSample:
        try:
            messages = extract_messages(data)
            input_ids, assistant_mask = tokenize_with_assistant_mask(self.tokenizer, messages)

            total_prompt_tokens = int(input_ids.shape[0])
            effective_length = min(total_prompt_tokens, max_tokens)
            input_ids = input_ids[:effective_length]
            assistant_mask = assistant_mask[:effective_length]

            return PreparedSample(
                original_data=data,
                input_ids=input_ids,
                assistant_mask=assistant_mask,
                total_prompt_tokens=total_prompt_tokens,
                prompt_tokens_used=effective_length,
            )
        except Exception as exc:
            return PreparedSample(
                original_data=data,
                input_ids=None,
                assistant_mask=None,
                total_prompt_tokens=0,
                prompt_tokens_used=0,
                error=str(exc),
            )

    @torch.inference_mode()
    def process_batch(
        self,
        batch_data: Sequence[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> List[WorkerResult]:
        prepared = [self._prepare_sample(item, max_tokens) for item in batch_data]
        results: List[Optional[WorkerResult]] = [None] * len(prepared)

        to_forward_indices = []
        for idx, sample in enumerate(prepared):
            if sample.error is not None:
                results[idx] = WorkerResult(
                    status="error", data={"status": "error", "error": sample.error}
                )
                continue
            if sample.prompt_tokens_used <= 1:
                results[idx] = WorkerResult(
                    status="success",
                    data={
                        "flatness": None,
                        "num_scored_tokens": 0,
                        "prompt_tokens_total": sample.total_prompt_tokens,
                        "prompt_tokens_used": sample.prompt_tokens_used,
                    },
                )
                continue
            to_forward_indices.append(idx)

        if not to_forward_indices:
            return [r for r in results if r is not None]

        max_len = max(prepared[idx].prompt_tokens_used for idx in to_forward_indices)
        batch_size = len(to_forward_indices)
        input_ids_batch = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask_batch = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )

        for row_idx, sample_idx in enumerate(to_forward_indices):
            sample = prepared[sample_idx]
            curr_len = sample.prompt_tokens_used
            input_ids_batch[row_idx, :curr_len] = sample.input_ids.to(self.device)
            attention_mask_batch[row_idx, :curr_len] = 1

        outputs = self.model_body(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state[:, :-1, :]

        all_hidden_chunks = []
        sample_token_counts = []
        for row_idx, sample_idx in enumerate(to_forward_indices):
            sample = prepared[sample_idx]
            curr_len = sample.prompt_tokens_used
            target_mask = sample.assistant_mask[1:curr_len].to(self.device)
            sample_hidden = hidden_states[row_idx, : curr_len - 1, :]
            selected_hidden = sample_hidden[target_mask]
            sample_token_counts.append(int(selected_hidden.shape[0]))
            if selected_hidden.shape[0] > 0:
                all_hidden_chunks.append(selected_hidden)

        if all_hidden_chunks:
            concat_hidden = torch.cat(all_hidden_chunks, dim=0)
            logits_chunk = self.lm_head(concat_hidden)
            all_flatness = flatness_from_logits(logits_chunk, temperature)
        else:
            all_flatness = torch.empty((0,), dtype=torch.float32, device=self.device)

        offset = 0
        for local_idx, sample_idx in enumerate(to_forward_indices):
            sample = prepared[sample_idx]
            token_count = sample_token_counts[local_idx]
            if token_count == 0:
                results[sample_idx] = WorkerResult(
                    status="success",
                    data={
                        "flatness": None,
                        "num_scored_tokens": 0,
                        "prompt_tokens_total": sample.total_prompt_tokens,
                        "prompt_tokens_used": sample.prompt_tokens_used,
                    },
                )
                continue

            token_flatness = all_flatness[offset : offset + token_count]
            offset += token_count
            results[sample_idx] = WorkerResult(
                status="success",
                data={
                    "flatness": float(token_flatness.mean().item()),
                    "num_scored_tokens": token_count,
                    "prompt_tokens_total": sample.total_prompt_tokens,
                    "prompt_tokens_used": sample.prompt_tokens_used,
                },
            )

        return [r for r in results if r is not None]


def pick_devices(num_workers: int) -> List[str]:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_workers > num_gpus:
            raise ValueError(
                f"Requested {num_workers} workers from --server-address, but only {num_gpus} CUDA devices are visible"
            )
        return [f"cuda:{idx}" for idx in range(num_workers)]
    if num_workers > 1:
        raise ValueError("Multiple workers require CUDA devices")
    return ["cpu"]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid json on line {line_idx}: {exc}") from exc
    return records


def main():
    args = parse_arguments()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be greater than 0")
    if not os.path.exists(args.input_file_path):
        raise ValueError(f"Input file does not exist: {args.input_file_path}")

    records = read_jsonl(args.input_file_path)
    output_file_path = default_output_path(args.input_file_path, args.output_dir_path)
    error_file_path = output_file_path.replace(".jsonl", "_error.jsonl")
    summary_file_path = output_file_path.replace(".jsonl", "_summary.json")

    worker_devices = pick_devices(args.num_workers)
    batch_size_per_worker = args.batch_size
    print("Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {output_file_path}")
    print(f"  Error file: {error_file_path}")
    print(f"  Summary file: {summary_file_path}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Batch size per worker: {batch_size_per_worker}")
    print(f"  Worker replicas: {len(worker_devices)}")
    print(f"  Worker devices: {worker_devices}")
    print("-" * 50)

    workers = [FlatnessWorker(args.model, device) for device in worker_devices]
    executor = ThreadPoolExecutor(max_workers=len(workers))
    pending = {}

    with (
        open(output_file_path, "w", encoding="utf-8") as output_handle,
        open(error_file_path, "w", encoding="utf-8") as error_handle,
        tqdm(total=len(records), desc="Processing") as pbar,
    ):
        next_record_idx = 0
        success_count = 0
        error_count = 0
        all_flatness_values: List[float] = []

        def submit(worker_idx: int, batch_records: List[Dict[str, Any]]):
            future = executor.submit(
                workers[worker_idx].process_batch,
                batch_records,
                args.max_tokens,
                args.temperature,
            )
            pending[future] = worker_idx

        def pop_next_batch() -> List[Dict[str, Any]]:
            nonlocal next_record_idx
            if next_record_idx >= len(records):
                return []
            end_idx = min(next_record_idx + batch_size_per_worker, len(records))
            batch = records[next_record_idx:end_idx]
            next_record_idx = end_idx
            return batch

        initial_jobs = min(len(workers), len(records))
        for worker_idx in range(initial_jobs):
            init_batch = pop_next_batch()
            if init_batch:
                submit(worker_idx, init_batch)

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                worker_idx = pending.pop(future)
                batch_results = future.result()
                for result in batch_results:
                    if result.status == "success":
                        output_handle.write(
                            json.dumps(result.data, ensure_ascii=False) + "\n"
                        )
                        success_count += 1
                        flatness_value = result.data.get("flatness")
                        if flatness_value is not None:
                            all_flatness_values.append(float(flatness_value))
                    else:
                        error_handle.write(json.dumps(result.data, ensure_ascii=False) + "\n")
                        error_count += 1
                    pbar.update(1)

                next_batch = pop_next_batch()
                if next_batch:
                    submit(worker_idx, next_batch)

    executor.shutdown(wait=True)
    half_comparison = build_half_comparison(all_flatness_values)
    with open(summary_file_path, "w", encoding="utf-8") as summary_handle:
        json.dump(half_comparison, summary_handle, ensure_ascii=False, indent=2)

    print("\nProcessing completed!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Valid flatness samples: {half_comparison['num_valid_samples']}")
    if half_comparison["num_valid_samples"] > 0:
        lower = half_comparison["lower_50_percent"]
        upper = half_comparison["upper_50_percent"]
        comp = half_comparison["comparison"]
        print("  Flatness split comparison (lower 50% vs upper 50%):")
        print(
            f"    Lower: count={lower['count']}, mean={lower['mean']:.6f}, median={lower['median']:.6f}, min={lower['min']:.6f}, max={lower['max']:.6f}"
        )
        print(
            f"    Upper: count={upper['count']}, mean={upper['mean']:.6f}, median={upper['median']:.6f}, min={upper['min']:.6f}, max={upper['max']:.6f}"
        )
        print(
            f"    Mean diff (upper-lower): {comp['mean_diff_upper_minus_lower']:.6f}"
        )
        ratio = comp["mean_ratio_upper_div_lower"]
        if ratio is None:
            print("    Mean ratio (upper/lower): undefined (lower mean is 0)")
        else:
            print(f"    Mean ratio (upper/lower): {ratio:.6f}")
    else:
        print("  Flatness split comparison skipped: no valid flatness values.")
    print(f"  Summary saved to: {summary_file_path}")


if __name__ == "__main__":
    main()
