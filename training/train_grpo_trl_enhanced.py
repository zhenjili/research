"""
GRPO Training with TRL - Enhanced Version for 8x A100 80GB

This script leverages TRL's optimizations:
- Native vLLM integration for 5-10x faster generation
- Liger Kernel for 20% throughput + 60% memory savings
- Accelerate for distributed training
- Custom execution-based reward function

Hardware Configuration:
    8x A100 80GB GPUs:
    - GPUs 0-1: vLLM generation (tensor parallel)
    - GPUs 2-7: DeepSpeed ZeRO-3 training

Usage:
    # Terminal 1: Start vLLM server
    CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.6

    # Terminal 2: Train with Accelerate + DeepSpeed
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \\
        --config_file configs/accelerate_trl_8xa100.yaml \\
        training/train_grpo_trl_enhanced.py \\
        --config configs/grpo_config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_rl_dataset(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Load RL prompts dataset from JSONL.

    Returns HuggingFace Dataset with columns:
    - prompt: The prompt text (TRL expects 'prompt' column)
    - test_cases: Test cases for reward computation
    - task_id: Unique identifier
    """
    data = []
    with open(data_path, "r") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if line.strip():
                item = json.loads(line)
                # TRL GRPOTrainer expects 'prompt' column
                data.append({
                    "prompt": item.get("prompt", ""),
                    "test_cases": item.get("test_cases", ""),
                    "task_id": item.get("task_id", f"task_{idx}"),
                    "entry_point": item.get("entry_point", ""),
                    "test_setup_code": item.get("test_setup_code", ""),
                })

    print(f"Loaded {len(data)} prompts from {data_path}")
    return Dataset.from_list(data)


def create_reward_function(config: Dict[str, Any], dataset: Dataset):
    """
    Create custom reward function for code execution.

    TRL GRPOTrainer expects: reward_fn(completions, prompts, **kwargs) -> List[float]
    Note: The order is (completions, prompts) not (prompts, completions)!
    """
    from rewards.reward_func import reward_func

    # Build a mapping from prompt to test_cases for quick lookup
    prompt_to_test_cases = {}
    for item in dataset:
        prompt_to_test_cases[item["prompt"]] = item["test_cases"]

    def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Compute execution-based rewards.

        Args:
            completions: List of generated code responses
            prompts: List of original prompts
            **kwargs: Additional context

        Returns:
            List of reward values (1.0 for pass, 0.0 for fail, -0.5 for errors)
        """
        # Get test cases for each prompt
        test_cases_list = []
        for prompt in prompts:
            # Try exact match first
            if prompt in prompt_to_test_cases:
                test_cases_list.append(prompt_to_test_cases[prompt])
            else:
                # Fallback: empty test cases (will return 0.0 reward)
                test_cases_list.append("")

        # Clean completions (remove special tokens)
        cleaned_completions = []
        for comp in completions:
            if "<|im_end|>" in comp:
                comp = comp.split("<|im_end|>")[0]
            cleaned_completions.append(comp)

        # Compute rewards using existing reward function
        rewards = reward_func(
            cleaned_completions,
            [""] * len(cleaned_completions),  # No ground truth needed
            test_cases_list,
        )

        # Convert to list of floats
        if torch.is_tensor(rewards):
            rewards = rewards.cpu().tolist()

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by accelerate)",
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    print("=" * 80)
    print("GRPO Training with TRL - Enhanced Version")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Local rank: {args.local_rank}")
    print()

    # Extract config sections
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})

    # Model settings
    model_name = model_cfg.get("pretrain", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = train_cfg.get("output_dir", "./outputs/qwen-grpo-trl")
    os.makedirs(output_dir, exist_ok=True)

    # Save config to output directory
    if args.local_rank <= 0:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    # =========================================================================
    # Load Tokenizer
    # =========================================================================
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print(f"Loading dataset: {data_cfg['prompt_data']}")
    dataset = load_rl_dataset(
        data_cfg["prompt_data"],
        max_samples=data_cfg.get("max_samples", None),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample prompt: {dataset[0]['prompt'][:100]}...")
    print()

    # =========================================================================
    # TRL GRPO Configuration
    # =========================================================================
    grpo_config = GRPOConfig(
        # Output directory
        output_dir=output_dir,

        # Mixed precision
        bf16=model_cfg.get("bf16", True),

        # Memory optimizations
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Use Liger Kernel for optimization
        use_liger_kernel=True,

        # Model initialization kwargs
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32,
            "trust_remote_code": model_cfg.get("trust_remote_code", True),
        },

        # GRPO algorithm settings
        num_generations=grpo_cfg.get("n_samples_per_prompt", 2),  # Samples per prompt
        num_iterations=1,  # GRPO uses single iteration
        beta=grpo_cfg.get("init_kl_coef", 0.001),  # KL coefficient
        epsilon=grpo_cfg.get("cliprange", 0.2),  # PPO clip range

        # Batch sizes
        per_device_train_batch_size=train_cfg.get("micro_train_batch_size", 1),
        gradient_accumulation_steps=max(1, train_cfg.get("train_batch_size", 2) // train_cfg.get("micro_train_batch_size", 1)),

        # Learning rate
        learning_rate=train_cfg.get("actor_learning_rate", 3e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),

        # Optimization
        adam_beta1=train_cfg.get("adam_betas", [0.9, 0.95])[0],
        adam_beta2=train_cfg.get("adam_betas", [0.9, 0.95])[1],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),

        # Training duration
        num_train_epochs=train_cfg.get("max_epochs", 1),
        max_steps=train_cfg.get("num_episodes", 2),

        # Generation settings
        max_completion_length=gen_cfg.get("generate_max_len", 512),
        temperature=gen_cfg.get("temperature", 1.0),
        top_p=gen_cfg.get("top_p", 0.95),

        # vLLM acceleration
        use_vllm=True,
        vllm_mode="server",  # Use external vLLM server
        vllm_server_host="0.0.0.0",
        vllm_server_port=8000,
        vllm_server_timeout=240.0,

        # Logging
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 1),
        save_total_limit=3,
        report_to=config.get("logging", {}).get("report_to", "tensorboard"),

        # Misc
        seed=data_cfg.get("seed", 42),
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,  # Keep all dataset columns
    )

    print("=" * 80)
    print("TRL GRPO Configuration:")
    print("=" * 80)
    print(f"  Model: {model_name}")
    print(f"  Use Liger Kernel: {grpo_config.use_liger_kernel}")
    print(f"  Use vLLM: {grpo_config.use_vllm}")
    print(f"  vLLM mode: {grpo_config.vllm_mode}")
    print(f"  Samples per prompt: {grpo_config.num_generations}")
    print(f"  Per-device batch size: {grpo_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {grpo_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  Max steps: {grpo_config.max_steps}")
    print(f"  Output dir: {output_dir}")
    print()

    # =========================================================================
    # Create Reward Function
    # =========================================================================
    print("Creating custom reward function...")
    reward_fn = create_reward_function(config, dataset)

    # =========================================================================
    # Initialize TRL GRPO Trainer
    # =========================================================================
    print("Initializing GRPO Trainer...")
    print(f"  Loading model: {model_name}")

    trainer = GRPOTrainer(
        model=model_name,  # Pass model name, TRL will load it
        args=grpo_config,  # Use 'args' instead of 'config'
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,  # Use 'reward_funcs' instead of 'reward_model'
    )

    print("Trainer initialized successfully!")
    print()

    # =========================================================================
    # Train
    # =========================================================================
    print("=" * 80)
    print("Starting Training...")
    print("=" * 80)
    print()

    try:
        trainer.train()
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # =========================================================================
    # Save Final Model
    # =========================================================================
    if args.local_rank <= 0:
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

        final_dir = os.path.join(output_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Final model saved to: {final_dir}")

        print("\nNext steps:")
        print(f"  1. Evaluate: python evaluation/run_evalplus_vllm.py --model_path {final_dir}")
        print(f"  2. Compare: python evaluation/compare_checkpoints.py --checkpoint_dir {output_dir}")


if __name__ == "__main__":
    main()
