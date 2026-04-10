"""Entropy / token-NLL sampling, plotting, and aggregate stats for regenerate_train_data."""

from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from tqdm import tqdm

CallSglang = Callable[..., Any]


def _line_sample_rng_seed(
    input_file_path: str, max_lines: int, sample_count: int
) -> int:
    """Stable seed from path + pool size + k (think vs non-think runs match)."""
    canonical = os.path.abspath(os.path.realpath(input_file_path))
    k = min(sample_count, max_lines)
    msg = f"{canonical}|max_lines={max_lines}|k={k}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def save_nll_curve(token_nlls: List[float], output_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(token_nlls) + 1), token_nlls, linewidth=1.2)
    plt.xlabel("Token index")
    plt.ylabel("NLL")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_per_position_mean_nll(
    sequences: List[List[float]],
) -> Tuple[List[float], List[int]]:
    """For each position i (0-based), mean NLL over sequences with length > i."""
    if not sequences:
        return [], []
    max_len = max(len(s) for s in sequences)
    sums = [0.0] * max_len
    counts = [0] * max_len
    for seq in sequences:
        for i, v in enumerate(seq):
            sums[i] += v
            counts[i] += 1
    means = [sums[i] / counts[i] for i in range(max_len)]
    return means, counts


def save_mean_nll_data(
    fig_dir: str,
    mean_nll: List[float],
    counts: List[int],
    num_sequences: int,
) -> None:
    payload = {
        "description": "Per-token mean NLL across sampled sequences; position k uses only samples with at least k tokens.",
        "num_sequences": num_sequences,
        "token_index": list(range(1, len(mean_nll) + 1)),
        "mean_nll": mean_nll,
        "count_per_position": counts,
    }
    path = os.path.join(fig_dir, "mean_nll.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_sample_conversation(
    conversations: List[Dict[str, Any]], output_path: str
) -> None:
    lines = []
    for idx, message in enumerate(conversations, start=1):
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        lines.append(f"[{idx}] {role}")
        lines.append(str(content))
        if "reasoning_content" in message and message["reasoning_content"] is not None:
            lines.append("")
            lines.append("[REASONING]")
            lines.append(str(message["reasoning_content"]))
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def load_random_jsonl_samples(
    input_file_path: str,
    sample_count: int,
    max_lines: int,
) -> List[Dict[str, Any]]:
    """Sample line indices with a deterministic RNG (same path/max_lines/k → same lines)."""
    k = min(sample_count, max_lines)
    seed = _line_sample_rng_seed(input_file_path, max_lines, sample_count)
    rng = random.Random(seed)
    selected_line_numbers = set(rng.sample(range(1, max_lines + 1), k=k))
    selected_samples = []
    with open(input_file_path, "r") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line_number > max_lines:
                break
            if line_number in selected_line_numbers:
                data = json.loads(line.strip())
                data["line_number"] = line_number
                selected_samples.append(data)
    return selected_samples


def _run_sample_nll_mode(
    args: Any,
    call_sglang: CallSglang,
    valid_server_addresses: List[str],
    max_lines: int,
) -> None:
    os.makedirs(args.fig_dir, exist_ok=True)
    selected_samples = load_random_jsonl_samples(
        args.input_file_path,
        sample_count=args.sample_num,
        max_lines=max_lines,
    )
    print(
        f"Sample-NLL mode: selected {len(selected_samples)} samples "
        f"from the first {max_lines} lines (deterministic per input path, max_lines, sample_num)."
    )
    if args.ignore_fig:
        print("  --ignore-fig: per-sample PNG/TXT will be skipped")

    executor = ThreadPoolExecutor(
        max_workers=min(
            args.concurrency * len(valid_server_addresses),
            max(1, len(selected_samples)),
        )
    )
    futures = []
    for idx, data in enumerate(selected_samples):
        server_address = valid_server_addresses[idx % len(valid_server_addresses)]
        futures.append(executor.submit(call_sglang, args, server_address, data))

    success_samples = 0
    error_samples = 0
    saved_figures = 0
    all_token_nll_sequences: List[List[float]] = []
    pbar_desc = "NLL sampling" if args.ignore_fig else "Plotting NLL curves"
    pbar = tqdm(total=len(futures), desc=pbar_desc)
    for future in futures:
        regen_data = future.result()
        if regen_data["status"] == "error":
            error_samples += 1
        else:
            success_samples += 1
            token_nlls = regen_data.get("token_nlls", [])
            line_number = regen_data.get("line_number")
            if token_nlls and line_number is not None:
                all_token_nll_sequences.append(token_nlls)
                if not args.ignore_fig:
                    output_path = os.path.join(args.fig_dir, f"{line_number}.png")
                    save_nll_curve(
                        token_nlls=token_nlls,
                        output_path=output_path,
                        title=f"Line {line_number} token-level NLL",
                    )
                    text_output_path = os.path.join(
                        args.fig_dir, f"{line_number}.txt"
                    )
                    save_sample_conversation(
                        conversations=regen_data.get("conversations", []),
                        output_path=text_output_path,
                    )
                    saved_figures += 1
        pbar.update(1)
    pbar.close()
    executor.shutdown(wait=True)

    if all_token_nll_sequences:
        mean_nll, pos_counts = compute_per_position_mean_nll(all_token_nll_sequences)
        save_mean_nll_data(
            args.fig_dir,
            mean_nll,
            pos_counts,
            num_sequences=len(all_token_nll_sequences),
        )
        mean_png = os.path.join(args.fig_dir, "mean.png")
        save_nll_curve(
            token_nlls=mean_nll,
            output_path=mean_png,
            title="Mean token-level NLL (per position, over sampled lines)",
        )
        print(f"  Mean NLL curve: {mean_png}")
        print(f"  Mean NLL data: {os.path.join(args.fig_dir, 'mean_nll.json')}")
    else:
        print("  No token NLL sequences collected; skipped mean.png / mean_nll.json")

    print(f"\nNLL sampling completed!")
    print(f"  Successful samples: {success_samples}")
    print(f"  Error samples: {error_samples}")
    if not args.ignore_fig:
        print(f"  Saved per-sample figures: {saved_figures}")
    print(f"  Output directory: {args.fig_dir}")


def _run_full_dataset_entropy(
    args: Any,
    call_sglang: CallSglang,
    valid_server_addresses: List[str],
    max_lines: int,
) -> None:
    per_sample_avg_losses = []
    token_counts = []
    success_samples = 0
    error_samples = 0

    def process_result(regen_data: Dict[str, Any]) -> None:
        nonlocal success_samples, error_samples
        if regen_data["status"] == "error":
            error_samples += 1
        else:
            success_samples += 1
            if "entropy" in regen_data:
                per_sample_avg_losses.append(regen_data["entropy"])
                token_counts.append(regen_data["num_tokens"])

    with open(args.input_file_path, "r") as input_file:
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {sa: [] for sa in valid_server_addresses}
        pbar = tqdm(total=max_lines, desc="Computing entropy")
        start_server_index = 0

        for line in input_file:
            if (
                args.num_samples is not None
                and success_samples + error_samples >= args.num_samples
            ):
                break

            data = json.loads(line.strip())
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            while len(waiting_queue[server_address]) >= args.concurrency:
                finished = False
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        process_result(req_future.result())
                        waiting_queue[server_address].remove(req_future)
                        finished = True
                if finished:
                    break

            req_future = executor.submit(call_sglang, args, server_address, data)
            waiting_queue[server_address].append(req_future)
            pbar.update(1)

        for sa, futures in waiting_queue.items():
            for f in futures:
                process_result(f.result())
        pbar.close()

    print(f"\nEntropy computation completed!")
    print(f"  Successful samples: {success_samples}")
    print(f"  Error samples: {error_samples}")
    if per_sample_avg_losses:
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts)
        avg_loss = sum(
            loss * num_tokens
            for loss, num_tokens in zip(per_sample_avg_losses, token_counts)
        ) / total_tokens
        print(f"  Samples with loss: {len(per_sample_avg_losses)}")
        print(f"  Total generated tokens: {total_tokens}")
        print(f"  Average tokens per sample: {avg_tokens:.1f}")
        print(f"  Average token loss / cross-entropy (nats): {avg_loss:.6f}")
        print(f"  Average token loss / cross-entropy (bits): {avg_loss / 0.693147:.6f}")
    else:
        print("  No valid entropy data collected.")


def compute_entropy_mode(args: Any, call_sglang: CallSglang) -> None:
    """Compute average sequence entropy, or sample-NLL mode with optional figures."""
    print(f"Entropy mode enabled — will compute sequence entropy only.")
    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  No think mode: {args.no_think}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Sample num: {args.sample_num}")
    print(f"  Ignore per-sample figures: {args.ignore_fig}")
    print(f"  Figure directory: {args.fig_dir}")
    print("-" * 50)

    total_lines = sum(1 for _ in open(args.input_file_path))
    max_lines = (
        min(total_lines, args.num_samples)
        if args.num_samples is not None
        else total_lines
    )

    if args.sample_num is not None:
        if args.sample_num <= 0:
            raise ValueError("--sample-num must be greater than 0")
        if not args.fig_dir:
            raise ValueError("--fig-dir is required when --sample-num is set")

    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(args, server_address, dummy_data, max_tokens=1)
        if result is not None:
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    if args.sample_num is not None:
        _run_sample_nll_mode(args, call_sglang, valid_server_addresses, max_lines)
        return

    _run_full_dataset_entropy(args, call_sglang, valid_server_addresses, max_lines)
