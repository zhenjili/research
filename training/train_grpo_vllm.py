"""
GRPO Training with vLLM for fast generation.
Uses vLLM for 5-10x faster batch inference.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    print(f"Loaded {len(prompts)} prompts from {data_path}")
    return prompts


def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compute_rewards(responses: List[str], test_cases: List[str], num_workers: int = 8) -> torch.Tensor:
    """Compute rewards by executing code in parallel."""
    from rewards.reward_func import reward_func
    from multiprocessing import Pool
    import numpy as np

    # Clean responses
    cleaned = []
    for resp in responses:
        if "<|im_end|>" in resp:
            resp = resp.split("<|im_end|>")[0]
        cleaned.append(resp)

    # Parallel execution with multiprocessing
    if len(cleaned) > num_workers:
        # Split into chunks for parallel processing
        chunk_size = (len(cleaned) + num_workers - 1) // num_workers
        chunks = []
        for i in range(0, len(cleaned), chunk_size):
            end_idx = min(i + chunk_size, len(cleaned))
            chunks.append((
                cleaned[i:end_idx],
                [""] * (end_idx - i),
                test_cases[i:end_idx]
            ))

        # Process chunks in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(reward_func, chunks)

        # Combine results
        all_rewards = []
        for r in results:
            if isinstance(r, torch.Tensor):
                all_rewards.extend(r.cpu().numpy())
            else:
                all_rewards.extend(r)
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
    else:
        # For small batches, use serial execution
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

        # Group normalization
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

    return advantages


def train_grpo_vllm(config: Dict[str, Any]):
    """Main GRPO training loop with vLLM."""

    print("=" * 60)
    print("GRPO Training with vLLM Acceleration")
    print("=" * 60)

    # Extract config
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})

    model_name = model_cfg.get("pretrain")
    output_dir = train_cfg.get("output_dir", "./outputs/qwen-grpo-vllm")

    n_samples_per_prompt = grpo_cfg.get("n_samples_per_prompt", 4)
    kl_coef = grpo_cfg.get("init_kl_coef", 0.001)
    cliprange = grpo_cfg.get("cliprange", 0.2)

    batch_size = train_cfg.get("train_batch_size", 4)
    num_episodes = train_cfg.get("num_episodes", 100)
    lr = train_cfg.get("actor_learning_rate", 3e-6)
    save_steps = train_cfg.get("save_steps", 10)
    logging_steps = train_cfg.get("logging_steps", 1)

    max_new_tokens = gen_cfg.get("generate_max_len", 512)
    temperature = gen_cfg.get("temperature", 1.0)
    top_p = gen_cfg.get("top_p", 0.95)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model for training (regular transformers)
    print(f"Loading policy model for training: {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Single GPU for training
        trust_remote_code=True,
    )

    if train_cfg.get("gradient_checkpointing", True):
        policy_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # NOTE: Reference model removed to save memory
    # GRPO can work without KL penalty by relying purely on reward-based advantages
    print("  Reference model: Not loaded (using reward-only GRPO)")

    # Initialize vLLM for fast generation
    # vLLM will automatically use available GPUs with tensor parallelism
    print(f"Initializing vLLM for generation: {model_name}")

    vllm_model = LLM(
        model=model_name,
        tensor_parallel_size=2,  # Use 2 GPUs for generation
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print("  vLLM initialized successfully!")

    # Load dataset
    prompts_data = load_prompts(data_cfg["prompt_data"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
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

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Samples per prompt: {n_samples_per_prompt}")
    print(f"  - Total samples per step: {batch_size * n_samples_per_prompt}")
    print(f"  - Learning rate: {lr}")
    print(f"  - vLLM for generation: GPU 1-2 (tensor_parallel_size=2)")
    print(f"  - Policy model for training: GPU 0")
    print(f"  - Reference model: None (memory-optimized GRPO)")

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        episode_start = time.time()
        print(f"\n[Episode {episode}] Starting...", flush=True)

        # Sample batch of prompts
        indices = np.random.choice(len(prompts_data), batch_size, replace=False)
        batch_prompts = [prompts_data[i] for i in indices]

        # Prepare prompts for vLLM (repeat each n_samples times)
        formatted_prompts = []
        test_cases_list = []
        for item in batch_prompts:
            formatted = format_prompt(item["prompt"])
            for _ in range(n_samples_per_prompt):
                formatted_prompts.append(formatted)
                test_cases_list.append(item["test_cases"])

        print(f"[Episode {episode}] Generating {len(formatted_prompts)} samples with vLLM...", flush=True)
        gen_start = time.time()

        # Generate with vLLM (FAST!)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        outputs = vllm_model.generate(formatted_prompts, sampling_params)

        # Extract completions
        completions = []
        for output in outputs:
            text = output.outputs[0].text
            completions.append(text)

        gen_time = time.time() - gen_start
        print(f"[Episode {episode}] Generated {len(completions)} samples in {gen_time:.2f}s ({len(completions)/gen_time:.1f} samples/s)", flush=True)

        # Compute rewards
        print(f"[Episode {episode}] Computing rewards...", flush=True)
        reward_start = time.time()
        rewards = compute_rewards(completions, test_cases_list)
        reward_time = time.time() - reward_start
        mean_reward = rewards.mean().item()
        print(f"[Episode {episode}] Rewards computed in {reward_time:.2f}s, mean: {mean_reward:.3f}", flush=True)

        # Compute advantages
        advantages = compute_grpo_advantages(rewards, n_samples_per_prompt)

        # ============================================================
        # GRPO Training Step - Policy Update with Gradient Accumulation
        # ============================================================
        print(f"[Episode {episode}] Running policy update...", flush=True)
        train_start = time.time()

        policy_model.train()
        optimizer.zero_grad()

        # Prepare full sequences
        full_sequences = []
        for prompt, completion in zip(formatted_prompts, completions):
            full_seq = prompt + completion
            full_sequences.append(full_seq)

        # Process in micro-batches to avoid OOM
        micro_batch_size = train_cfg.get("micro_train_batch_size", 2)
        num_samples = len(full_sequences)
        num_micro_batches = (num_samples + micro_batch_size - 1) // micro_batch_size

        print(f"[Episode {episode}] Processing {num_samples} samples in {num_micro_batches} micro-batches of size {micro_batch_size}", flush=True)

        total_policy_loss = 0.0
        total_entropy = 0.0
        all_log_probs = []

        for micro_batch_idx in range(num_micro_batches):
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, num_samples)

            micro_full_sequences = full_sequences[start_idx:end_idx]
            micro_formatted_prompts = formatted_prompts[start_idx:end_idx]
            micro_advantages = advantages[start_idx:end_idx]

            # Tokenize
            encodings = tokenizer(
                micro_full_sequences,
                padding=True,
                truncation=True,
                max_length=gen_cfg.get("prompt_max_len", 512) + max_new_tokens,
                return_tensors="pt",
            ).to(device)

            prompt_encodings = tokenizer(
                micro_formatted_prompts,
                padding=True,
                truncation=True,
                max_length=gen_cfg.get("prompt_max_len", 512),
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

            # Sum log probs for this micro-batch
            sequence_log_probs = (token_log_probs * completion_mask.float()).sum(dim=1)
            all_log_probs.append(sequence_log_probs.detach())

            # GRPO loss for this micro-batch
            micro_advantages = micro_advantages.to(device)
            policy_loss = -(micro_advantages * sequence_log_probs).mean()

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

            # Scale loss by number of micro-batches for gradient accumulation
            scaled_loss = total_loss / num_micro_batches

            # Backward pass (accumulate gradients)
            scaled_loss.backward()

            # Accumulate metrics
            total_policy_loss += policy_loss.item() / num_micro_batches
            total_entropy += entropy_bonus.item() / num_micro_batches

            # Free memory
            del encodings, prompt_encodings, logits, shift_logits, shift_labels
            del log_probs, token_log_probs, sequence_log_probs
            torch.cuda.empty_cache()

        # Gradient clipping
        max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        train_time = time.time() - train_start

        # Logging
        if episode % logging_steps == 0:
            all_log_probs_tensor = torch.cat(all_log_probs)
            print(f"[Episode {episode}] Training metrics:", flush=True)
            print(f"  - Policy loss: {total_policy_loss:.4f}", flush=True)
            print(f"  - Entropy: {total_entropy:.4f}", flush=True)
            print(f"  - Mean advantage: {advantages.mean().item():.4f}", flush=True)
            print(f"  - Mean log prob: {all_log_probs_tensor.mean().item():.4f}", flush=True)
            print(f"  - Learning rate: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            print(f"  - Training time: {train_time:.2f}s", flush=True)
            print(f"  - Micro-batches: {num_micro_batches}", flush=True)

        episode_time = time.time() - episode_start
        print(f"[Episode {episode}] Complete in {episode_time:.2f}s (gen: {gen_time:.2f}s, reward: {reward_time:.2f}s, train: {train_time:.2f}s)", flush=True)

        # Save checkpoint
        if (episode + 1) % save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{episode+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[Episode {episode}] Saved checkpoint to {checkpoint_dir}")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train_grpo_vllm(config)
