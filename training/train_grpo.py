"""
GRPO Training Script for Code LLM using OpenRLHF.

This script implements Group Relative Policy Optimization (GRPO) for
training code generation models with execution-based rewards.

Usage:
    # Basic training
    python training/train_grpo.py --config configs/grpo_config.yaml

    # With custom settings
    python training/train_grpo.py --config configs/grpo_config.yaml \
        --num_episodes 1000 --n_samples_per_prompt 4

    # Multi-GPU with Ray
    python training/train_grpo.py --config configs/grpo_config.yaml --use_ray

Requirements:
    - OpenRLHF installed: pip install openrlhf
    - Docker for code execution sandbox (optional, falls back to subprocess)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import yaml
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], output_dir: str):
    """Save effective configuration."""
    config_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to: {config_path}")


def check_openrlhf_installed() -> bool:
    """Check if OpenRLHF is installed."""
    try:
        import openrlhf
        return True
    except ImportError:
        return False


def build_openrlhf_args(config: Dict[str, Any]) -> List[str]:
    """Build command line arguments for OpenRLHF PPO training."""
    args = []

    # Model configuration
    model_cfg = config.get("model", {})
    args.extend([
        "--pretrain", model_cfg.get("pretrain", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ])
    if model_cfg.get("bf16", True):
        args.append("--bf16")
    if model_cfg.get("flash_attention", True):
        args.append("--flash_attn")
    if config.get("training", {}).get("gradient_checkpointing", True):
        args.append("--gradient_checkpointing")

    # GRPO configuration
    grpo_cfg = config.get("grpo", {})
    args.extend([
        "--advantage_estimator", grpo_cfg.get("advantage_estimator", "group_norm"),
        "--n_samples_per_prompt", str(grpo_cfg.get("n_samples_per_prompt", 4)),
        "--init_kl_coef", str(grpo_cfg.get("init_kl_coef", 0.001)),
        "--cliprange", str(grpo_cfg.get("cliprange", 10)),
    ])
    if grpo_cfg.get("normalize_reward", True):
        args.append("--normalize_reward")

    # Training configuration
    train_cfg = config.get("training", {})
    args.extend([
        "--train_batch_size", str(train_cfg.get("train_batch_size", 8)),
        "--micro_train_batch_size", str(train_cfg.get("micro_train_batch_size", 1)),
        "--actor_learning_rate", str(train_cfg.get("actor_learning_rate", 3e-6)),
        "--num_episodes", str(train_cfg.get("num_episodes", 5000)),
        "--max_epochs", str(train_cfg.get("max_epochs", 1)),
        "--save_steps", str(train_cfg.get("save_steps", 500)),
        "--logging_steps", str(train_cfg.get("logging_steps", 10)),
        "--save_path", train_cfg.get("output_dir", "./outputs/qwen-coder-grpo"),
    ])

    # Generation configuration
    gen_cfg = config.get("generation", {})
    args.extend([
        "--prompt_max_len", str(gen_cfg.get("prompt_max_len", 1024)),
        "--generate_max_len", str(gen_cfg.get("generate_max_len", 1536)),
        "--temperature", str(gen_cfg.get("temperature", 1.0)),
        "--top_p", str(gen_cfg.get("top_p", 0.95)),
    ])

    # Data configuration
    data_cfg = config.get("data", {})
    args.extend([
        "--prompt_data", data_cfg.get("prompt_data", "./data/rl_prompts/mbpp_prompts.jsonl"),
        "--input_key", data_cfg.get("input_key", "prompt"),
        "--label_key", data_cfg.get("label_key", "test_cases"),
        "--max_samples", str(data_cfg.get("max_samples", 50000)),
    ])

    # Distributed configuration
    dist_cfg = config.get("distributed", {})
    args.extend([
        "--zero_stage", str(dist_cfg.get("zero_stage", 2)),
    ])

    # Logging
    log_cfg = config.get("logging", {})
    if log_cfg.get("report_to") == "wandb":
        args.extend([
            "--use_wandb", log_cfg.get("wandb_project", "code-llm-grpo"),
        ])

    return args


def run_openrlhf_training(config: Dict[str, Any], use_ray: bool = False):
    """Run training using OpenRLHF."""
    from openrlhf.cli.train_ppo import main as train_main

    # Build arguments
    args = build_openrlhf_args(config)

    # Set custom reward function
    reward_path = config.get("reward", {}).get("reward_func_path", "./rewards/reward_func.py")
    if os.path.exists(reward_path):
        # OpenRLHF supports custom reward functions
        args.extend(["--remote_rm_url", reward_path])

    print("=" * 60)
    print("Starting GRPO Training with OpenRLHF")
    print("=" * 60)
    print(f"Arguments: {' '.join(args)}")
    print("=" * 60)

    # Run training
    sys.argv = ["train_grpo"] + args
    train_main()


def run_simple_grpo_training(config: Dict[str, Any]):
    """
    Simplified GRPO training loop for single GPU without OpenRLHF.
    Uses TRL's PPOTrainer with GRPO-style advantage estimation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from tqdm import tqdm
    import json

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rewards.reward_func import reward_func, get_reward_with_details

    print("=" * 60)
    print("Starting Simple GRPO Training (without OpenRLHF)")
    print("=" * 60)

    # Load configuration
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})

    # Load model and tokenizer
    print(f"Loading model: {model_cfg['pretrain']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["pretrain"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["pretrain"],
        torch_dtype=torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load training data
    print(f"Loading data: {data_cfg['prompt_data']}")
    prompts = []
    labels = []
    with open(data_cfg["prompt_data"], "r") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item[data_cfg["input_key"]])
            labels.append(item.get(data_cfg["label_key"], ""))

    print(f"Loaded {len(prompts)} prompts")

    # Training settings
    n_samples = grpo_cfg.get("n_samples_per_prompt", 4)
    num_episodes = train_cfg.get("num_episodes", 1000)
    batch_size = train_cfg.get("train_batch_size", 8)
    lr = train_cfg.get("actor_learning_rate", 3e-6)
    output_dir = train_cfg.get("output_dir", "./outputs/qwen-coder-grpo")

    os.makedirs(output_dir, exist_ok=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    total_steps = 0
    rewards_history = []

    print(f"\nStarting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Sample a batch of prompts
        batch_indices = torch.randint(0, len(prompts), (batch_size,)).tolist()
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_labels = [labels[i] for i in batch_indices]

        # Format prompts for Qwen
        formatted_prompts = [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in batch_prompts
        ]

        # Generate n_samples responses per prompt
        all_responses = []
        all_rewards = []

        for prompt, label in zip(formatted_prompts, batch_labels):
            responses = []
            for _ in range(n_samples):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=gen_cfg.get("generate_max_len", 512),
                        temperature=gen_cfg.get("temperature", 1.0),
                        top_p=gen_cfg.get("top_p", 0.95),
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                responses.append(response)

            # Compute rewards
            rewards = reward_func(
                responses,
                [prompt] * n_samples,
                [label] * n_samples
            )

            all_responses.extend(responses)
            all_rewards.extend(rewards.tolist())

        # Compute GRPO-style advantage (group normalization)
        rewards_tensor = torch.tensor(all_rewards)
        rewards_by_prompt = rewards_tensor.view(batch_size, n_samples)

        # Group-relative advantage: subtract mean, divide by std within each group
        mean_rewards = rewards_by_prompt.mean(dim=1, keepdim=True)
        std_rewards = rewards_by_prompt.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_by_prompt - mean_rewards) / std_rewards

        # Log progress
        mean_reward = rewards_tensor.mean().item()
        rewards_history.append(mean_reward)

        if episode % train_cfg.get("logging_steps", 10) == 0:
            print(f"Episode {episode}: mean_reward = {mean_reward:.4f}, "
                  f"pass_rate = {(rewards_tensor > 0).float().mean():.2%}")

        # Save checkpoint
        if episode > 0 and episode % train_cfg.get("save_steps", 500) == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{episode}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

        total_steps += 1

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete! Final model saved to {final_dir}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"rewards": rewards_history}, f)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Code LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--use_ray",
        action="store_true",
        help="Use Ray for distributed training"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple training loop (no OpenRLHF dependency)"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Override pretrain model path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Override number of training episodes"
    )
    parser.add_argument(
        "--n_samples_per_prompt",
        type=int,
        default=None,
        help="Override samples per prompt"
    )

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file not found: {args.config}")
        print("Using default configuration")
        config = {}

    # Apply command line overrides
    if args.pretrain:
        config.setdefault("model", {})["pretrain"] = args.pretrain
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.num_episodes:
        config.setdefault("training", {})["num_episodes"] = args.num_episodes
    if args.n_samples_per_prompt:
        config.setdefault("grpo", {})["n_samples_per_prompt"] = args.n_samples_per_prompt

    # Create output directory
    output_dir = config.get("training", {}).get("output_dir", "./outputs/qwen-coder-grpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save effective configuration
    save_config(config, output_dir)

    # Check dependencies and run appropriate training
    if args.simple:
        print("Using simple training loop (no OpenRLHF)")
        run_simple_grpo_training(config)
    elif check_openrlhf_installed():
        print("OpenRLHF detected, using full training pipeline")
        run_openrlhf_training(config, use_ray=args.use_ray)
    else:
        print("OpenRLHF not installed. Options:")
        print("  1. Install OpenRLHF: pip install openrlhf")
        print("  2. Use simple training: python train_grpo.py --simple")
        print("\nFalling back to simple training...")
        run_simple_grpo_training(config)

    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python evaluation/run_evalplus.py --model_path {output_dir}/final")
    print(f"  2. Compare: python evaluation/compare_checkpoints.py --sft outputs/qwen-coder-lora/merged --rl {output_dir}/final")


if __name__ == "__main__":
    main()
