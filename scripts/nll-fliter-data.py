"""
Filter an existing regenerated chat dataset by sample NLL.

This script reads a jsonl dataset whose items already contain assistant replies,
scores all assistant tokens with a single SGLang prefill-only request
(`/generate` with `max_new_tokens=0` and `return_logprob=True`), computes the
mean NLL over all assistant tokens in each sample, and keeps the highest-NLL
50 percent of successfully scored samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REQUEST_TIMEOUT = 600


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Score existing assistant replies by NLL and keep the top 50%"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help=(
            "Concurrent requests per server. Total concurrency is this value times "
            "the number of valid server addresses."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help=(
            "Accepted for CLI compatibility with regenerate_train_data.py. "
            "Scoring uses prefill only, so no new tokens are generated."
        ),
    )
    parser.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        required=True,
        help="Server address and port for sglang model server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Accepted for CLI compatibility. Prefill scoring is deterministic; this "
            "value is sent to the backend but does not affect max_new_tokens=0 scoring."
        ),
    )
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
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


def normalize_messages(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Input item does not contain a valid `conversations` list")

    normalized = []
    for message in conversations:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        normalized_message = {"role": role}
        content = message.get("content", "")
        if isinstance(content, (str, list)):
            normalized_message["content"] = content
        else:
            normalized_message["content"] = extract_text(content)

        if "reasoning_content" in message:
            normalized_message["reasoning_content"] = message["reasoning_content"]
        if "tool_calls" in message:
            normalized_message["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            normalized_message["tool_call_id"] = message["tool_call_id"]

        normalized.append(normalized_message)

    if not normalized:
        raise ValueError("No valid chat messages found in sample")
    return normalized


def simplify_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    simplified = []
    for message in messages:
        simplified.append(
            {
                "role": str(message.get("role")),
                "content": extract_text(message.get("content")),
            }
        )
    return simplified


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
    messages: Sequence[Dict[str, Any]],
) -> Tuple[List[int], List[bool]]:
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
        input_ids_tensor = tokenized["input_ids"][0]
        assistant_mask_tensor = tokenized.get("assistant_masks")
        if assistant_mask_tensor is None:
            assistant_mask_tensor = tokenized.get("assistant_tokens_mask")
        if assistant_mask_tensor is None:
            raise KeyError("assistant token mask was not returned")

        assistant_mask_tensor = assistant_mask_tensor[0].to(dtype=torch.bool)
        if has_assistant_message and int(assistant_mask_tensor.sum().item()) == 0:
            raise ValueError("assistant token mask is empty; fallback reconstruction")

        return input_ids_tensor.tolist(), assistant_mask_tensor.tolist()
    except Exception:
        simplified_messages = simplify_messages(messages)
        input_ids = tokenizer.apply_chat_template(
            simplified_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        assistant_mask = build_assistant_mask_fallback(tokenizer, simplified_messages)
        if len(assistant_mask) != len(input_ids):
            raise ValueError(
                f"assistant mask length mismatch: {len(assistant_mask)} vs {len(input_ids)}"
            )
        return list(input_ids), [bool(x) for x in assistant_mask]


def validate_server(server_address: str) -> bool:
    payload = {
        "text": "hello",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        },
    }
    try:
        response = requests.post(
            f"http://{server_address}/generate",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def extract_logprob_value(entry: Any) -> float | None:
    current = entry
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]

    if isinstance(current, (list, tuple)) and current:
        first = current[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, (list, tuple)):
            return extract_logprob_value(first)
    return None


def score_sample(
    args,
    server_address: str,
    tokenizer,
    data: Dict[str, Any],
    line_number: int,
) -> Dict[str, Any]:
    try:
        messages = normalize_messages(data)
        input_ids, assistant_mask = tokenize_with_assistant_mask(tokenizer, messages)

        assistant_positions = [
            idx for idx, is_assistant_token in enumerate(assistant_mask) if is_assistant_token
        ]
        if not assistant_positions:
            raise ValueError("No assistant tokens found in sample")

        first_assistant_position = assistant_positions[0]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": 0,
            },
            "return_logprob": True,
            "logprob_start_len": first_assistant_position,
            "return_text_in_logprobs": True,
            "stream": False,
        }

        response = requests.post(
            f"http://{server_address}/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        response_json = response.json()

        meta_info = response_json.get("meta_info", {})
        input_token_logprobs = meta_info.get("input_token_logprobs")
        if not isinstance(input_token_logprobs, list) or not input_token_logprobs:
            raise ValueError("Backend did not return input_token_logprobs")

        returned_suffix_start = len(input_ids) - len(input_token_logprobs)
        if returned_suffix_start < first_assistant_position:
            raise ValueError(
                "Returned prompt logprobs are shorter than requested assistant suffix"
            )

        token_nlls = []
        for relative_idx, entry in enumerate(input_token_logprobs):
            prompt_idx = returned_suffix_start + relative_idx
            if prompt_idx >= len(assistant_mask) or not assistant_mask[prompt_idx]:
                continue

            logprob = extract_logprob_value(entry)
            if logprob is None:
                continue
            token_nlls.append(-logprob)

        if not token_nlls:
            raise ValueError("No assistant-token logprobs were collected")

        sample_nll = sum(token_nlls) / len(token_nlls)
        return {
            "status": "success",
            "line_number": line_number,
            "sample_nll": sample_nll,
            "num_scored_tokens": len(token_nlls),
            "data": data,
        }
    except Exception as exc:
        return {
            "status": "error",
            "line_number": line_number,
            "error": str(exc),
            "data": data,
        }


def default_error_path(output_file_path: str) -> str:
    output_path = Path(output_file_path)
    if output_path.suffix == ".jsonl":
        return str(output_path.with_name(f"{output_path.stem}_error.jsonl"))
    return f"{output_file_path}_error.jsonl"


def process_completed_future(
    future,
    successful_results: List[Dict[str, Any]],
    error_results: List[Dict[str, Any]],
    pbar,
) -> None:
    result = future.result()
    if result["status"] == "success":
        successful_results.append(result)
    else:
        error_results.append(result)
    pbar.update(1)


def main():
    args = parse_arguments()

    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    if args.concurrency <= 0:
        raise ValueError("Concurrency must be greater than 0")

    print("Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print("  Keep rule: highest sample-NLL 50% of successfully scored samples")
    print("-" * 50)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    total_lines = sum(1 for _ in open(args.input_file_path, "r"))
    if total_lines == 0:
        raise ValueError("Input file is empty")

    valid_server_addresses = []
    for server_address in args.server_address:
        if validate_server(server_address):
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if not valid_server_addresses:
        raise ValueError("No server address is available")

    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    max_workers = args.concurrency * len(valid_server_addresses)
    successful_results: List[Dict[str, Any]] = []
    error_results: List[Dict[str, Any]] = []
    pending_futures = set()
    next_server_idx = 0

    with open(args.input_file_path, "r") as input_file, ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        pbar = tqdm(total=total_lines, desc="Scoring sample NLL")
        for line_number, line in enumerate(input_file, start=1):
            data = json.loads(line.strip())
            server_address = valid_server_addresses[next_server_idx]
            next_server_idx = (next_server_idx + 1) % len(valid_server_addresses)

            pending_futures.add(
                executor.submit(score_sample, args, server_address, tokenizer, data, line_number)
            )

            if len(pending_futures) >= max_workers:
                done, pending_futures = wait(
                    pending_futures,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    process_completed_future(
                        future,
                        successful_results,
                        error_results,
                        pbar,
                    )

        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                process_completed_future(
                    future,
                    successful_results,
                    error_results,
                    pbar,
                )
        pbar.close()

    if not successful_results:
        raise ValueError("No samples were scored successfully")

    keep_count = math.ceil(len(successful_results) * 0.5)
    ranked_results = sorted(
        successful_results,
        key=lambda item: (-item["sample_nll"], item["line_number"]),
    )
    selected_line_numbers = {
        item["line_number"] for item in ranked_results[:keep_count]
    }
    kept_results_in_original_order = [
        item
        for item in sorted(successful_results, key=lambda result: result["line_number"])
        if item["line_number"] in selected_line_numbers
    ]

    output_parent = os.path.dirname(os.path.abspath(args.output_file_path))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    error_file_path = default_error_path(args.output_file_path)

    with (
        open(args.output_file_path, "w") as output_file,
        open(error_file_path, "w") as error_file,
    ):
        for item in kept_results_in_original_order:
            output_file.write(json.dumps(item["data"], ensure_ascii=False) + "\n")
        for item in sorted(error_results, key=lambda result: result["line_number"]):
            error_payload = {
                "status": "error",
                "line_number": item["line_number"],
                "error": item["error"],
                "id": item["data"].get("id"),
            }
            error_file.write(json.dumps(error_payload, ensure_ascii=False) + "\n")

    kept_nlls = [item["sample_nll"] for item in ranked_results[:keep_count]]
    dropped_nlls = [item["sample_nll"] for item in ranked_results[keep_count:]]
    threshold_nll = kept_nlls[-1]
    mean_tokens = statistics.fmean(
        item["num_scored_tokens"] for item in successful_results
    )

    print("\nFiltering completed!")
    print(f"  Total input samples: {total_lines}")
    print(f"  Successful samples: {len(successful_results)}")
    print(f"  Error samples: {len(error_results)}")
    print(f"  Kept samples: {len(kept_results_in_original_order)}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Error file: {error_file_path}")
    print(f"  Mean scored assistant tokens per sample: {mean_tokens:.2f}")
    print(f"  Selected NLL threshold (smallest kept sample-NLL): {threshold_nll:.6f}")
    print(f"  Mean kept sample-NLL: {statistics.fmean(kept_nlls):.6f}")
    if dropped_nlls:
        print(f"  Mean dropped sample-NLL: {statistics.fmean(dropped_nlls):.6f}")


if __name__ == "__main__":
    main()
