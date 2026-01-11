"""
GRPO Training with vLLM and Multi-GPU DDP support.
Uses DDP for policy model training and vLLM for fast generation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(data_path: str, max_samples: int = None):
    """Load prompts from JSONL file."""
    prompts = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    if max_samples:
        prompts = prompts[:max_samples]

    return prompts


def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compute_rewards(responses: List[str], test_cases: List[str], num_workers: int = 8) -> torch.Tensor:
    """Compute rewards by executing code in parallel."""
    from rewards.reward_func import reward_func
    from multiprocessing import Pool

    # Clean responses
    cleaned = []
    for resp in responses:
        if "<|im_end|>" in resp:
            resp = resp.split("<|im_end|>")[0]
        cleaned.append(resp)

    # Parallel execution with multiprocessing
    if len(cleaned) > num_workers:
        chunk_size = (len(cleaned) + num_workers - 1) // num_workers
        chunks = []
        for i in range(0, len(cleaned), chunk_size):
            end_idx = min(i + chunk_size, len(cleaned))
            chunks.append((
                cleaned[i:end_idx],
                [""] * (end_idx - i),
                test_cases[i:end_idx]
            ))

        with Pool(processes=num_workers) as pool:
            results = pool.starmap(reward_func, chunks)

        all_rewards = []
        for r in results:
            if isinstance(r, torch.Tensor):
                all_rewards.extend(r.cpu().numpy())
            else:
                all_rewards.extend(r)
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
    else:
        rewards = reward_func(cleaned, [""] * len(cleaned), test_cases)

    return rewards


def compute_grpo_advantages(rewards: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Compute group-relative advantages for GRPO."""
    batch_size = len(rewards) // n_samples
    advantages = torch.zeros_like(rewards)

    for i in range(batch_size):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        group_rewards = rewards[start_idx:end_idx]

        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

    return advantages


def train_grpo_ddp(config: Dict[str, Any]):
    """Main GRPO training loop with DDP."""

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print("=" * 60)
        print("GRPO Training with Multi-GPU DDP + vLLM")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")

    # Extract config
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})

    model_name = model_cfg.get("pretrain")
    output_dir = train_cfg.get("output_dir", "./outputs/qwen-grpo-ddp")

    n_samples_per_prompt = grpo_cfg.get("n_samples_per_prompt", 4)
    batch_size = train_cfg.get("train_batch_size", 4)
    num_episodes = train_cfg.get("num_episodes", 100)
    lr = train_cfg.get("actor_learning_rate", 3e-6)
    save_steps = train_cfg.get("save_steps", 10)
    logging_steps = train_cfg.get("logging_steps", 1)

    max_new_tokens = gen_cfg.get("generate_max_len", 512)
    temperature = gen_cfg.get("temperature", 1.0)
    top_p = gen_cfg.get("top_p", 0.95)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    # Load tokenizer
    if is_main_process:
        print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model for training
    if is_main_process:
        print(f"Loading policy model on {world_size} GPUs with DDP")

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if train_cfg.get("gradient_checkpointing", True):
        policy_model.gradient_checkpointing_enable()
        if is_main_process:
            print("  Gradient checkpointing enabled")

    # Wrap with DDP
    if world_size > 1:
        policy_model = DDP(
            policy_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if is_main_process:
            print(f"  Model wrapped with DDP across {world_size} GPUs")

    # Initialize vLLM for generation (only on rank 0)
    # For simplicity, rank 0 will use the SAME GPUs for both policy and generation
    # This avoids GPU assignment conflicts
    vllm_model = None
    if is_main_process:
        print(f"Initializing vLLM for generation (using GPU 0 for vLLM)")
        vllm_model = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Single GPU for vLLM to avoid conflicts
            gpu_memory_utilization=0.3,  # Lower utilization since sharing with policy
            trust_remote_code=True,
            dtype="bfloat16",
            disable_log_stats=True,
        )

        print("  vLLM initialized successfully!")

    # Load dataset
    prompts_data = load_prompts(data_cfg["prompt_data"])
    if is_main_process:
        print(f"Loaded {len(prompts_data)} prompts")

    # Optimizer (only trainable parameters)
    model_params = policy_model.module.parameters() if isinstance(policy_model, DDP) else policy_model.parameters()
    optimizer = torch.optim.AdamW(
        model_params,
        lr=lr,
        betas=tuple(train_cfg.get("adam_betas", [0.9, 0.95])),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(num_episodes * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_episodes,
    )

    if is_main_process:
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"  - Batch size per GPU: {batch_size // world_size}")
        print(f"  - Total batch size: {batch_size}")
        print(f"  - Samples per prompt: {n_samples_per_prompt}")
        print(f"  - Total samples per step: {batch_size * n_samples_per_prompt}")
        print(f"  - Learning rate: {lr}")
        print(f"  - DDP GPUs: {list(range(world_size))}")
        if vllm_model:
            print(f"  - vLLM GPUs: {vllm_gpus}")

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training", disable=not is_main_process):
        episode_start = time.time()

        if is_main_process:
            print(f"\n[Episode {episode}] Starting...", flush=True)

        # Only rank 0 generates samples
        if is_main_process:
            # Sample batch of prompts
            indices = np.random.choice(len(prompts_data), batch_size, replace=False)
            batch_prompts = [prompts_data[i] for i in indices]

            # Prepare prompts for vLLM
            formatted_prompts = []
            test_cases_list = []
            for item in batch_prompts:
                formatted = format_prompt(item["prompt"])
                for _ in range(n_samples_per_prompt):
                    formatted_prompts.append(formatted)
                    test_cases_list.append(item["test_cases"])

            print(f"[Episode {episode}] Generating {len(formatted_prompts)} samples with vLLM...", flush=True)
            gen_start = time.time()

            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|im_end|>", "<|im_start|>"],
            )

            outputs = vllm_model.generate(formatted_prompts, sampling_params)
            completions = [output.outputs[0].text for output in outputs]

            gen_time = time.time() - gen_start
            print(f"[Episode {episode}] Generated {len(completions)} samples in {gen_time:.2f}s ({len(completions)/gen_time:.1f} samples/s)", flush=True)

            # Compute rewards
            print(f"[Episode {episode}] Computing rewards...", flush=True)
            reward_start = time.time()
            rewards = compute_rewards(completions, test_cases_list, num_workers=16)
            reward_time = time.time() - reward_start
            mean_reward = rewards.mean().item()
            print(f"[Episode {episode}] Rewards computed in {reward_time:.2f}s, mean: {mean_reward:.3f}", flush=True)

            # Compute advantages
            advantages = compute_grpo_advantages(rewards, n_samples_per_prompt)
        else:
            # Other ranks wait for data from rank 0
            formatted_prompts = None
            completions = None
            advantages = None
            gen_time = 0
            reward_time = 0

        # Broadcast data to all ranks
        if world_size > 1:
            # Broadcast via pickle (simpler for complex data structures)
            if is_main_process:
                data_to_broadcast = {
                    'formatted_prompts': formatted_prompts,
                    'completions': completions,
                    'advantages': advantages,
                    'gen_time': gen_time,
                    'reward_time': reward_time,
                }
            else:
                data_to_broadcast = None

            # Use object list for broadcasting
            data_list = [data_to_broadcast]
            dist.broadcast_object_list(data_list, src=0)

            if not is_main_process:
                formatted_prompts = data_list[0]['formatted_prompts']
                completions = data_list[0]['completions']
                advantages = data_list[0]['advantages']
                gen_time = data_list[0]['gen_time']
                reward_time = data_list[0]['reward_time']

        # Policy update (all ranks participate)
        if is_main_process:
            print(f"[Episode {episode}] Running policy update on {world_size} GPUs...", flush=True)

        train_start = time.time()

        policy_model.train()
        optimizer.zero_grad()

        # Prepare sequences
        full_sequences = []
        for prompt, completion in zip(formatted_prompts, completions):
            full_sequences.append(prompt + completion)

        # Split data across GPUs
        samples_per_gpu = len(full_sequences) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else len(full_sequences)

        local_sequences = full_sequences[start_idx:end_idx]
        local_prompts = formatted_prompts[start_idx:end_idx]
        local_advantages = advantages[start_idx:end_idx].to(device)

        # Tokenize
        encodings = tokenizer(
            local_sequences,
            padding=True,
            truncation=True,
            max_length=gen_cfg.get("prompt_max_len", 768) + max_new_tokens,
            return_tensors="pt",
        ).to(device)

        prompt_encodings = tokenizer(
            local_prompts,
            padding=True,
            truncation=True,
            max_length=gen_cfg.get("prompt_max_len", 768),
            return_tensors="pt",
        ).to(device)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = policy_model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
            )
            logits = outputs.logits

        # Compute log probabilities
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = encodings.input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask prompt tokens
        prompt_lens = prompt_encodings.attention_mask.sum(dim=1)
        completion_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, plen in enumerate(prompt_lens):
            completion_start = max(0, plen.item() - 1)
            completion_mask[i, completion_start:] = True
        completion_mask = completion_mask & (shift_labels != tokenizer.pad_token_id)

        sequence_log_probs = (token_log_probs * completion_mask.float()).sum(dim=1)

        # GRPO loss
        policy_loss = -(local_advantages * sequence_log_probs).mean()

        # Entropy
        entropy_coef = grpo_cfg.get("entropy_coef", 0.01)
        if entropy_coef > 0:
            probs = F.softmax(shift_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropy_masked = (entropy * completion_mask.float()).sum(dim=1)
            entropy_bonus = entropy_masked.mean()
            total_loss = policy_loss - entropy_coef * entropy_bonus
        else:
            total_loss = policy_loss
            entropy_bonus = torch.tensor(0.0)

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        train_time = time.time() - train_start

        # Gather metrics from all ranks
        if world_size > 1:
            policy_loss_tensor = torch.tensor([policy_loss.item()], device=device)
            entropy_tensor = torch.tensor([entropy_bonus.item()], device=device)

            dist.all_reduce(policy_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(entropy_tensor, op=dist.ReduceOp.AVG)

            avg_policy_loss = policy_loss_tensor.item()
            avg_entropy = entropy_tensor.item()
        else:
            avg_policy_loss = policy_loss.item()
            avg_entropy = entropy_bonus.item()

        # Logging (only main process)
        if is_main_process and episode % logging_steps == 0:
            print(f"[Episode {episode}] Training metrics:", flush=True)
            print(f"  - Policy loss: {avg_policy_loss:.4f}", flush=True)
            print(f"  - Entropy: {avg_entropy:.4f}", flush=True)
            print(f"  - Mean advantage: {advantages.mean().item():.4f}", flush=True)
            print(f"  - Mean log prob: {sequence_log_probs.mean().item():.4f}", flush=True)
            print(f"  - Learning rate: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            print(f"  - Training time: {train_time:.2f}s", flush=True)

        episode_time = time.time() - episode_start

        if is_main_process:
            print(f"[Episode {episode}] Complete in {episode_time:.2f}s (gen: {gen_time:.2f}s, reward: {reward_time:.2f}s, train: {train_time:.2f}s)", flush=True)

        # Save checkpoint (only main process)
        if is_main_process and (episode + 1) % save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{episode+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the underlying model (not DDP wrapper)
            model_to_save = policy_model.module if isinstance(policy_model, DDP) else policy_model
            model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[Episode {episode}] Saved checkpoint to {checkpoint_dir}")

    if is_main_process:
        print("\nTraining complete!")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    train_grpo_ddp(config)
