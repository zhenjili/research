#!/usr/bin/env python3
"""
GRPO training with DeepSpeed ZeRO and vLLM generation.
Uses DeepSpeed for distributed training to avoid conflicts with vLLM.
"""

import os
import sys
import yaml
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import deepspeed
from multiprocessing import Pool

# Add research directory to path for reward function
RESEARCH_DIR = "/lambda/nfs/jiex/code/research"
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from rewards.reward_func import reward_func


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


def compute_rewards(responses: List[str], test_cases: List[str], num_workers: int = 8) -> torch.Tensor:
    """Compute rewards by executing code in parallel."""
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


def train_grpo_deepspeed(config: Dict[str, Any], local_rank: int):
    """Main GRPO training loop with DeepSpeed."""

    # Extract config
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    grpo_cfg = config.get("grpo", {})

    model_name = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    lr = train_cfg.get("actor_learning_rate", 3e-6)
    num_episodes = train_cfg.get("num_episodes", 50)
    train_batch_size = train_cfg.get("train_batch_size", 8)
    micro_batch_size = train_cfg.get("micro_train_batch_size", 2)
    n_samples = grpo_cfg.get("n_samples_per_prompt", 12)
    entropy_coef = grpo_cfg.get("entropy_coef", 0.01)

    prompt_max_len = gen_cfg.get("prompt_max_len", 768)
    generate_max_len = gen_cfg.get("generate_max_len", 768)
    temperature = gen_cfg.get("temperature", 0.7)
    top_p = gen_cfg.get("top_p", 0.9)

    save_steps = train_cfg.get("save_steps", 5)
    save_dir = train_cfg.get("save_dir", "./checkpoints/grpo_vllm_deepspeed")

    is_main = local_rank == 0

    if is_main:
        print("=" * 60)
        print("GRPO Training with DeepSpeed ZeRO + vLLM")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Episodes: {num_episodes}")
        print(f"Batch size: {train_batch_size}")
        print(f"Micro batch size: {micro_batch_size}")
        print(f"Samples per prompt: {n_samples}")
        print(f"Learning rate: {lr}")
        print(f"Entropy coefficient: {entropy_coef}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Loading tokenizer: {model_name}")

    # Load policy model
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if is_main:
        print(f"Loading policy model with DeepSpeed ZeRO-2")

    # Initialize vLLM for generation (only on rank 0)
    vllm_model = None
    if is_main:
        print(f"Initializing vLLM for generation")
        # vLLM will use GPUs 6-7
        os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
        vllm_model = LLM(
            model=model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.5,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_log_stats=True,
        )
        # Restore CUDA devices for training
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        print("  vLLM initialized successfully!")

    # Load dataset
    prompts_data = load_prompts(data_cfg["prompt_data"])
    if is_main:
        print(f"Loaded {len(prompts_data)} prompts")

    # Create optimizer before DeepSpeed init (required for ZeRO-2)
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=lr,
        betas=tuple(train_cfg.get("adam_betas", [0.9, 0.95])),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Prepare DeepSpeed config
    ds_config = {
        "train_batch_size": train_batch_size * n_samples,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": (train_batch_size * n_samples) // (micro_batch_size * torch.cuda.device_count()),
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=policy_model,
        optimizer=optimizer,
        config=ds_config,
    )

    if is_main:
        print(f"DeepSpeed initialized with ZeRO-2")
        print(f"World size: {torch.distributed.get_world_size()}")
        os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for episode in range(num_episodes):
        episode_start_time = time.time()

        if is_main:
            print(f"\n{'='*60}")
            print(f"Episode {episode}/{num_episodes}")
            print(f"{'='*60}")

        # Use same random seed across all ranks for consistent prompt sampling
        random.seed(episode + 42)
        batch_prompts = random.sample(prompts_data, min(train_batch_size, len(prompts_data)))

        # Only rank 0 does generation and reward computation
        # Then saves tokenized data to shared file for other ranks to load
        if is_main:
            gen_start = time.time()
            print(f"Generating {n_samples} samples per prompt...")

            # Expand prompts: each prompt gets n_samples
            expanded_prompts = []
            test_cases_list = []
            for item in batch_prompts:
                prompt_text = item["prompt"]
                test_case = item.get("test_cases", "")
                for _ in range(n_samples):
                    expanded_prompts.append(prompt_text)
                    test_cases_list.append(test_case)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=generate_max_len,
                n=1,
            )

            outputs = vllm_model.generate(expanded_prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            gen_time = time.time() - gen_start
            print(f"  Generated {len(responses)} responses in {gen_time:.2f}s")

            # Compute rewards
            reward_start = time.time()
            print(f"Computing rewards with 8 parallel workers...")
            rewards = compute_rewards(responses, test_cases_list, num_workers=8)
            reward_time = time.time() - reward_start
            print(f"  Rewards computed in {reward_time:.2f}s")
            print(f"  Mean reward: {rewards.mean().item():.4f}")

            # Compute advantages
            advantages = compute_grpo_advantages(rewards, n_samples)

            # Prepare full sequences for training
            full_sequences = []
            for prompt, response in zip(expanded_prompts, responses):
                full_seq = prompt + response
                full_sequences.append(full_seq)

            # Tokenize all sequences once on rank 0
            all_encodings = tokenizer(
                full_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_len + generate_max_len,
            )

            # Save to shared file for other ranks
            shared_data = {
                'input_ids': all_encodings.input_ids,
                'attention_mask': all_encodings.attention_mask,
                'advantages': advantages,
                'rewards': rewards,
            }
            shared_path = f"/tmp/grpo_episode_{episode}_data.pt"
            torch.save(shared_data, shared_path)

        # Synchronize all ranks - wait for rank 0 to finish
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks load the shared data
        shared_path = f"/tmp/grpo_episode_{episode}_data.pt"
        shared_data = torch.load(shared_path)
        all_input_ids = shared_data['input_ids']
        all_attention_mask = shared_data['attention_mask']
        advantages = shared_data['advantages']
        rewards = shared_data['rewards']

        # Training step
        if is_main:
            train_start = time.time()
            print(f"Training policy model...")

        model_engine.train()

        # Split data across GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        total_samples = all_input_ids.size(0)
        samples_per_gpu = total_samples // world_size
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples

        local_input_ids = all_input_ids[start_idx:end_idx].to(f'cuda:{local_rank}')
        local_attention_mask = all_attention_mask[start_idx:end_idx].to(f'cuda:{local_rank}')
        local_advantages = advantages[start_idx:end_idx].to(f'cuda:{local_rank}')

        # Compute loss in micro-batches
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        num_local_samples = local_input_ids.size(0)
        num_micro_batches = (num_local_samples + micro_batch_size - 1) // micro_batch_size

        for micro_idx in range(num_micro_batches):
            micro_start = micro_idx * micro_batch_size
            micro_end = min(micro_start + micro_batch_size, num_local_samples)

            batch_input_ids = local_input_ids[micro_start:micro_end]
            batch_attention_mask = local_attention_mask[micro_start:micro_end]
            batch_advs = local_advantages[micro_start:micro_end]

            # Forward pass
            outputs = model_engine(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            logits = outputs.logits

            # Compute log probs for generated tokens only
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_input_ids[:, 1:].contiguous()

            # Calculate per-token log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs,
                2,
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask padding tokens
            mask = (shift_labels != tokenizer.pad_token_id).float()
            masked_log_probs = selected_log_probs * mask

            # Sum log probs per sequence
            seq_log_probs = masked_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Policy loss with advantages
            policy_loss = -(seq_log_probs * batch_advs).mean()

            # Entropy bonus
            probs = F.softmax(shift_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + entropy_coef * entropy_loss

            # Backward
            model_engine.backward(loss)

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.mean().item()

            # Clean up
            del logits, shift_logits, shift_labels, log_probs
            torch.cuda.empty_cache()

        # Optimizer step (DeepSpeed handles gradient accumulation)
        model_engine.step()

        if is_main:
            train_time = time.time() - train_start
            episode_time = time.time() - episode_start_time

            avg_loss = total_loss / num_micro_batches
            avg_policy_loss = total_policy_loss / num_micro_batches
            avg_entropy = total_entropy / num_micro_batches

            print(f"\n  Episode {episode} Summary:")
            print(f"    Loss: {avg_loss:.4f}")
            print(f"    Policy Loss: {avg_policy_loss:.4f}")
            print(f"    Entropy: {avg_entropy:.4f}")
            print(f"    Mean Reward: {rewards.mean().item():.4f}")
            print(f"  Timing:")
            print(f"    Generation: {gen_time:.2f}s")
            print(f"    Rewards: {reward_time:.2f}s")
            print(f"    Training: {train_time:.2f}s")
            print(f"    Total: {episode_time:.2f}s")

        # Save checkpoint
        if (episode + 1) % save_steps == 0 and is_main:
            ckpt_path = Path(save_dir) / f"episode_{episode+1}"
            model_engine.save_checkpoint(str(ckpt_path))
            print(f"\n  Checkpoint saved to {ckpt_path}")

    if is_main:
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_grpo_deepspeed(config, args.local_rank)


if __name__ == "__main__":
    main()
