"""
GRPO Training Script - Standalone Implementation.

This implements Group Relative Policy Optimization (GRPO) similar to DeepSeek-R1.
GRPO uses group-relative advantage estimation instead of a learned critic.

Usage:
    python training/train_grpo_trl.py --config configs/grpo_config.yaml

For A10 24GB GPU:
    python training/train_grpo_trl.py --config configs/grpo_config.yaml
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PromptDataset(Dataset):
    """Dataset for RL prompts with test cases."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)

        print(f"Loaded {len(self.data)} prompts from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item.get("prompt", ""),
            "test_cases": item.get("test_cases", ""),
            "task_id": item.get("task_id", f"task_{idx}"),
        }


def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compute_rewards(
    responses: List[str],
    test_cases: List[str],
) -> torch.Tensor:
    """Compute rewards by executing generated code against test cases."""
    from rewards.reward_func import reward_func

    # Clean up responses
    cleaned_responses = []
    for resp in responses:
        # Remove special tokens
        if "<|im_end|>" in resp:
            resp = resp.split("<|im_end|>")[0]
        cleaned_responses.append(resp)

    # Compute rewards
    rewards = reward_func(
        cleaned_responses,
        [""] * len(cleaned_responses),
        test_cases,
    )

    return rewards


def compute_grpo_advantages(
    rewards: torch.Tensor,
    n_samples_per_prompt: int,
) -> torch.Tensor:
    """
    Compute GRPO-style group-relative advantages.

    For each prompt, we have n_samples responses. The advantage is computed
    as (reward - mean) / std within each group.
    """
    batch_size = rewards.shape[0] // n_samples_per_prompt

    # Reshape to (batch_size, n_samples)
    rewards_grouped = rewards.view(batch_size, n_samples_per_prompt)

    # Compute group statistics
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

    # Normalize within each group
    advantages = (rewards_grouped - mean) / std

    # Flatten back
    return advantages.view(-1)


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute log probabilities for the given sequences."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out padding
    mask = (shift_labels != -100).float()

    # Sum log probs per sequence
    seq_log_probs = (token_log_probs * mask).sum(dim=-1)

    return seq_log_probs


def compute_policy_loss(
    model: nn.Module,
    ref_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float,
    cliprange: float,
    micro_batch_size: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Compute GRPO policy loss with KL penalty using gradient accumulation.

    Args:
        model: The policy model being trained
        ref_model: The reference model (frozen, may be on CPU)
        input_ids: Full sequences (prompt + response)
        attention_mask: Attention mask
        response_mask: Mask for response tokens only
        advantages: GRPO advantages
        kl_coef: KL divergence coefficient
        cliprange: PPO clip range
        micro_batch_size: Batch size for gradient accumulation

    Returns:
        Dictionary with loss and metrics
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    total_loss = torch.tensor(0.0, device=device)
    total_pg_loss = torch.tensor(0.0, device=device)
    total_kl_loss = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    total_ratio = torch.tensor(0.0, device=device)

    for i in range(num_micro_batches):
        start_idx = i * micro_batch_size
        end_idx = min((i + 1) * micro_batch_size, batch_size)

        mb_input_ids = input_ids[start_idx:end_idx]
        mb_attention_mask = attention_mask[start_idx:end_idx]
        mb_response_mask = response_mask[start_idx:end_idx]
        mb_advantages = advantages[start_idx:end_idx]

        # Get logits from policy model
        outputs = model(input_ids=mb_input_ids, attention_mask=mb_attention_mask)
        logits = outputs.logits

        # Get reference logits (model may be on CPU)
        with torch.no_grad():
            ref_input_ids = mb_input_ids.to(ref_model.device)
            ref_attention_mask = mb_attention_mask.to(ref_model.device)
            ref_outputs = ref_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)
            ref_logits = ref_outputs.logits.to(device)

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = mb_input_ids[..., 1:].contiguous()
        shift_response_mask = mb_response_mask[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Compute per-token KL divergence
        kl_div = token_log_probs - ref_token_log_probs

        # Apply response mask
        masked_log_probs = token_log_probs * shift_response_mask
        masked_kl = kl_div * shift_response_mask

        # Sum over sequence
        seq_log_probs = masked_log_probs.sum(dim=-1)
        seq_kl = masked_kl.sum(dim=-1) / (shift_response_mask.sum(dim=-1) + 1e-8)

        # Compute ratio for clipping with numerical stability
        with torch.no_grad():
            old_log_probs = ref_token_log_probs * shift_response_mask
            old_seq_log_probs = old_log_probs.sum(dim=-1)

        # Clamp log ratio to prevent exp overflow
        log_ratio = seq_log_probs - old_seq_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Check for NaN and skip if found
        if torch.isnan(pg_loss) or torch.isinf(pg_loss):
            print(f"Warning: NaN/Inf detected in pg_loss, skipping micro-batch")
            continue

        # KL penalty
        kl_loss = kl_coef * seq_kl.mean()

        # Micro-batch loss
        mb_loss = (pg_loss + kl_loss) / num_micro_batches

        # Accumulate gradients
        mb_loss.backward()

        # Track metrics
        with torch.no_grad():
            total_loss += mb_loss
            total_pg_loss += pg_loss / num_micro_batches
            total_kl_loss += kl_loss / num_micro_batches
            total_kl += seq_kl.mean() / num_micro_batches
            total_ratio += ratio.mean() / num_micro_batches

    return {
        "loss": total_loss,
        "pg_loss": total_pg_loss,
        "kl_loss": total_kl_loss,
        "kl": total_kl,
        "ratio": total_ratio,
    }


def train_grpo(config: Dict[str, Any]):
    """Main GRPO training loop."""

    print("=" * 60)
    print("GRPO Training - Standalone Implementation")
    print("=" * 60)

    # Extract config sections
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})

    # Model settings
    model_name = model_cfg.get("pretrain", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = train_cfg.get("output_dir", "./outputs/qwen-grpo")

    # GRPO settings
    n_samples_per_prompt = grpo_cfg.get("n_samples_per_prompt", 4)
    kl_coef = grpo_cfg.get("init_kl_coef", 0.001)
    cliprange = grpo_cfg.get("cliprange", 0.2)

    # Training settings
    batch_size = train_cfg.get("train_batch_size", 4)
    num_episodes = train_cfg.get("num_episodes", 1000)
    lr = train_cfg.get("actor_learning_rate", 5e-7)
    save_steps = train_cfg.get("save_steps", 500)
    logging_steps = train_cfg.get("logging_steps", 10)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # Generation settings
    max_new_tokens = gen_cfg.get("generate_max_len", 512)
    temperature = gen_cfg.get("temperature", 1.0)
    top_p = gen_cfg.get("top_p", 0.95)
    prompt_max_len = gen_cfg.get("prompt_max_len", 1024)

    os.makedirs(output_dir, exist_ok=True)

    # Device
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

    # Load policy model with gradient checkpointing for memory efficiency
    print(f"Loading policy model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",  # Avoid flash attention for gradient checkpointing
    )
    # Enable gradient checkpointing for memory savings
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    model.train()

    # Load reference model (frozen) - use CPU offloading to save GPU memory
    print(f"Loading reference model: {model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32,
        trust_remote_code=True,
        device_map="cpu",  # Keep on CPU to save GPU memory
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load dataset
    print(f"Loading dataset: {data_cfg['prompt_data']}")
    dataset = PromptDataset(
        data_cfg["prompt_data"],
        tokenizer,
        max_length=prompt_max_len,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(train_cfg.get("adam_betas", [0.9, 0.95])[0],
               train_cfg.get("adam_betas", [0.9, 0.95])[1]),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    warmup_steps = int(num_episodes * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_episodes,
    )

    # Training loop
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Samples per prompt: {n_samples_per_prompt}")
    print(f"  - Total samples per step: {batch_size * n_samples_per_prompt}")
    print(f"  - Learning rate: {lr}")
    print(f"  - KL coefficient: {kl_coef}")
    print(f"  - Clip range: {cliprange}")

    reward_history = []
    best_reward = -float("inf")

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for episode in tqdm(range(num_episodes), desc="Training"):
        try:
            print(f"\n[Episode {episode}] Starting...", flush=True)
            # Sample a batch of prompts
            batch_indices = torch.randint(0, len(dataset), (batch_size,)).tolist()
            print(f"[Episode {episode}] Sampled {len(batch_indices)} prompts", flush=True)

            all_input_ids = []
            all_attention_masks = []
            all_response_masks = []
            all_test_cases = []
            decoded_responses = []

            # Generate n_samples responses per prompt
            model.eval()
            generation_failed = False
            print(f"[Episode {episode}] Starting generation for {batch_size} prompts × {n_samples_per_prompt} samples = {batch_size * n_samples_per_prompt} total", flush=True)
            with torch.no_grad():
                for prompt_idx, idx in enumerate(batch_indices):
                    print(f"[Episode {episode}] Generating for prompt {prompt_idx+1}/{len(batch_indices)}", flush=True)
                    if generation_failed:
                        break

                    item = dataset[idx]
                    formatted_prompt = format_prompt(item["prompt"])
                    test_cases = item["test_cases"]

                    # Tokenize prompt
                    prompt_encoded = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=prompt_max_len,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(device)
                    prompt_len = prompt_ids.shape[1]

                    for sample_idx in range(n_samples_per_prompt):
                        print(f"[Episode {episode}] Prompt {prompt_idx+1}/{len(batch_indices)}, Sample {sample_idx+1}/{n_samples_per_prompt} - Generating...", flush=True)
                        # Generate response with error handling
                        try:
                            import time
                            gen_start = time.time()
                            output_ids = model.generate(
                                prompt_ids,
                                **generation_kwargs,
                            )
                            gen_time = time.time() - gen_start
                            print(f"[Episode {episode}] Sample generated in {gen_time:.2f}s", flush=True)
                        except RuntimeError as e:
                            print(f"\nWarning: Generation failed: {e}")
                            generation_failed = True
                            # Try to reset CUDA state
                            torch.cuda.empty_cache()
                            break

                        # Get full sequence
                        full_ids = output_ids[0]
                        full_len = full_ids.shape[0]

                        # Create attention mask
                        attention_mask = torch.ones(full_len, device=device)

                        # Create response mask (1 for response tokens, 0 for prompt)
                        response_mask = torch.zeros(full_len, device=device)
                        response_mask[prompt_len:] = 1.0

                        all_input_ids.append(full_ids)
                        all_attention_masks.append(attention_mask)
                        all_response_masks.append(response_mask)
                        all_test_cases.append(test_cases)

                        # Decode for reward computation
                        response_text = tokenizer.decode(
                            full_ids[prompt_len:],
                            skip_special_tokens=False,
                        )
                        decoded_responses.append(response_text)

            # Skip this episode if generation failed
            if generation_failed or len(all_input_ids) == 0:
                print(f"\nSkipping episode {episode} due to generation failure")
                continue

            print(f"[Episode {episode}] Generation complete! Computing rewards for {len(decoded_responses)} samples...", flush=True)
            # Compute rewards
            import time
            reward_start = time.time()
            rewards = compute_rewards(decoded_responses, all_test_cases)
            reward_time = time.time() - reward_start
            print(f"[Episode {episode}] Rewards computed in {reward_time:.2f}s, mean reward: {rewards.mean().item():.3f}", flush=True)

            # Compute GRPO advantages
            advantages = compute_grpo_advantages(rewards, n_samples_per_prompt)

            # Pad sequences for batch processing
            max_len = max(ids.shape[0] for ids in all_input_ids)
            padded_input_ids = []
            padded_attention_masks = []
            padded_response_masks = []

            for ids, attn, resp in zip(all_input_ids, all_attention_masks, all_response_masks):
                pad_len = max_len - ids.shape[0]
                if pad_len > 0:
                    ids = F.pad(ids, (0, pad_len), value=tokenizer.pad_token_id)
                    attn = F.pad(attn, (0, pad_len), value=0)
                    resp = F.pad(resp, (0, pad_len), value=0)
                padded_input_ids.append(ids)
                padded_attention_masks.append(attn)
                padded_response_masks.append(resp)

            batch_input_ids = torch.stack(padded_input_ids)
            batch_attention_masks = torch.stack(padded_attention_masks)
            batch_response_masks = torch.stack(padded_response_masks)
            batch_advantages = advantages.to(device)

            # Policy gradient update with gradient accumulation
            model.train()
            optimizer.zero_grad()

            # Micro batch size for memory-efficient training
            micro_batch_size = train_cfg.get("micro_train_batch_size", 2)

            loss_dict = compute_policy_loss(
                model=model,
                ref_model=ref_model,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                response_mask=batch_response_masks,
                advantages=batch_advantages,
                kl_coef=kl_coef,
                cliprange=cliprange,
                micro_batch_size=micro_batch_size,
            )

            loss = loss_dict["loss"]
            # Note: backward() already called in compute_policy_loss

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            # Logging
            mean_reward = rewards.mean().item()
            pass_rate = (rewards > 0).float().mean().item()
            reward_history.append(mean_reward)

            if episode % logging_steps == 0:
                print(f"\nEpisode {episode}:")
                print(f"  Mean reward: {mean_reward:.4f}")
                print(f"  Pass rate: {pass_rate:.2%}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  KL: {loss_dict['kl'].item():.4f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            if episode > 0 and episode % save_steps == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{episode}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_dir = os.path.join(output_dir, "best")
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

        except Exception as e:
            print(f"\nError in episode {episode}: {e}")
            torch.cuda.empty_cache()
            continue

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete! Final model saved to {final_dir}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"rewards": reward_history}, f)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Save config
    output_dir = config.get("training", {}).get("output_dir", "./outputs/qwen-grpo")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Run training
    train_grpo(config)

    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python evaluation/run_evalplus_vllm.py --model_path {output_dir}/final")


if __name__ == "__main__":
    main()
