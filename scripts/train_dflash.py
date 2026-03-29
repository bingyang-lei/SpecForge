#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import json
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import concatenate_datasets, load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def _parse_jsonl_paths(path_arg: str) -> list[str]:
    """Parse a single jsonl path or a JSON-encoded path list."""
    path_arg = path_arg.strip()
    if path_arg.startswith("["):
        try:
            parsed = json.loads(path_arg)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Invalid JSON list for --train-data-path: {path_arg}"
            ) from err
        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError("--train-data-path list must be a non-empty JSON array.")
        if not all(isinstance(p, str) and p.strip() for p in parsed):
            raise ValueError(
                "--train-data-path list must contain non-empty string file paths."
            )
        return [p.strip() for p in parsed]

    return [path_arg]


def _prepare_train_data_file(train_data_path_arg: str, cache_dir: str) -> str:
    """
    Return the train data file path.

    If multiple jsonl paths are provided as a JSON list, merge them into a single
    cached jsonl file before training.
    """
    import hashlib

    train_paths = _parse_jsonl_paths(train_data_path_arg)
    for path in train_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Train data file not found: {path}")

    if len(train_paths) == 1:
        return train_paths[0]

    file_signatures = []
    for path in train_paths:
        stat = os.stat(path)
        file_signatures.append(f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}")
    merged_key = hashlib.md5("|".join(file_signatures).encode()).hexdigest()
    merged_dir = os.path.join(cache_dir, "merged_jsonl")
    merged_path = os.path.join(merged_dir, f"train_merged_{merged_key}.jsonl")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        os.makedirs(merged_dir, exist_ok=True)
        if not os.path.exists(merged_path):
            print_on_rank0(
                f"Merging {len(train_paths)} train jsonl files into: {merged_path}"
            )
            with open(merged_path, "w", encoding="utf-8") as fout:
                for src in train_paths:
                    last_line = None
                    with open(src, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                            last_line = line
                    if last_line is not None and not last_line.endswith("\n"):
                        fout.write("\n")
        else:
            print_on_rank0(f"Reusing merged train jsonl file: {merged_path}")

    if is_dist:
        dist.barrier()

    if not os.path.isfile(merged_path):
        raise RuntimeError(f"Failed to create merged train data file: {merged_path}")

    return merged_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--train-data-path",
        type=str,
<<<<<<< HEAD
        nargs="+",
        required=True,
        help="Training data path(s). Supports one or multiple json/jsonl files, "
        'or a single JSON list string (e.g. \'["a.jsonl","b.jsonl"]\').',
=======
        required=True,
        help="Single jsonl path, or a JSON-encoded list of jsonl paths. "
        "When a list is provided, files are merged before training.",
>>>>>>> a130d7b (temp: collect dflash local changes)
    )
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory of the checkpoint to resume training from",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_models(args) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def _normalize_data_paths(data_paths_arg, arg_name: str) -> list[str]:
    """Normalize data path arg into a non-empty list of file paths."""
    if isinstance(data_paths_arg, str):
        paths = [data_paths_arg]
    else:
        paths = list(data_paths_arg)

    # Backward-compatible: allow JSON list string passed as a single CLI argument.
    if len(paths) == 1:
        maybe_json_list = paths[0].strip()
        if maybe_json_list.startswith("[") and maybe_json_list.endswith("]"):
            try:
                parsed = json.loads(maybe_json_list)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON list for --{arg_name}: {maybe_json_list}"
                ) from e
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) for item in parsed
            ):
                raise ValueError(
                    f"--{arg_name} JSON list must be list[str], got: {parsed}"
                )
            paths = parsed

    paths = [p for p in paths if isinstance(p, str) and p.strip()]
    if not paths:
        raise ValueError(f"--{arg_name} must contain at least one valid path")
    return paths


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    train_data_paths = _normalize_data_paths(args.train_data_path, "train-data-path")
    cache_params_string = (
        f"{'|'.join(train_data_paths)}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    if len(train_data_paths) == 1:
        train_dataset = load_dataset("json", data_files=train_data_paths[0])["train"]
    else:
        train_datasets = [
            load_dataset("json", data_files=data_path)["train"]
            for data_path in train_data_paths
        ]
        train_dataset = concatenate_datasets(train_datasets)
        print_on_rank0(
            f"Merged {len(train_data_paths)} train datasets into "
            f"{len(train_dataset)} samples"
        )

    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )

    tracker.log(logdict, step=global_step)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    target_model, draft_model = build_models(args)

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Using checkpoint: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided ckpt dir {args.ckpt_dir} is not a valid directory."
            )

    if args.resume and os.path.isdir(args.output_dir):
        last_checkpoint_result = get_last_checkpoint(args.output_dir, prefix="epoch")
        draft_model_last_checkpoint = last_checkpoint_result[0]
        if len(last_checkpoint_result) > 1 and last_checkpoint_result[1] is not None:
            ckpt_info = last_checkpoint_result[1]
        if draft_model_last_checkpoint is not None:
            print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        else:
            print_on_rank0(
                "No existing checkpoint found under output-dir, start from scratch."
            )

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_rank0(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",  # Adjust if Qwen/Llama differs
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
    )

    dflash_model = FSDP(
        dflash_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]
    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(f"Restored scheduler, lr={optimizer.get_learning_rate():.6f}")

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.hidden_states.cuda()  # Ensure on GPU

            loss, accuracy = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dflash_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
