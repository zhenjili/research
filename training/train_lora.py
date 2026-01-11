"""
LoRA Fine-tuning Script for Qwen2.5-Coder-7B

Usage:
    python training/train_lora.py --data_path data/oss_instruct_75k.json
    python training/train_lora.py --config configs/lora_config.yaml
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    trust_remote_code: bool = True
    use_flash_attention: bool = True


@dataclass
class LoRAArguments:
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataArguments:
    data_path: str = None
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 8


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_instruction(example: dict) -> str:
    """Format instruction data for Qwen chat template."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_content = instruction

    # Qwen chat format
    formatted = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

    return formatted


def load_and_prepare_dataset(data_path: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load and prepare dataset for training."""

    # Load JSON data
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(data_path, split="train")

    print(f"Loaded {len(dataset)} examples")

    # Format and tokenize
    def tokenize_function(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                "output": examples["output"][i],
            }
            texts.append(format_instruction(example))

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized_dataset


def create_model_and_tokenizer(model_args: ModelArguments, lora_args: LoRAArguments):
    """Create model and tokenizer with LoRA configuration."""

    print(f"Loading model: {model_args.model_name_or_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading configuration
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # Enable flash attention if available
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Code LLM")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--wandb_project", type=str, default="code-llm-training", help="W&B project name")

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)

    # Override with command line arguments
    data_path = args.data_path or config.get("data_path", "data/oss_instruct_75k.json")
    output_dir = args.output_dir or config.get("output_dir", "./outputs/qwen-coder-lora")
    num_epochs = args.num_epochs or config.get("num_train_epochs", 3)
    batch_size = args.batch_size or config.get("per_device_train_batch_size", 4)
    learning_rate = args.learning_rate or config.get("learning_rate", 2e-4)

    # Initialize arguments
    model_args = ModelArguments(
        model_name_or_path=config.get("model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    )

    lora_args = LoRAArguments(
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_dropout=config.get("lora_dropout", 0.05),
    )

    data_args = DataArguments(
        data_path=data_path,
        max_seq_length=config.get("max_seq_length", 2048),
    )

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_args, lora_args)

    # Load dataset
    train_dataset = load_and_prepare_dataset(
        data_args.data_path,
        tokenizer,
        data_args.max_seq_length,
    )

    # Split for validation
    split_dataset = train_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=learning_rate,
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=config.get("eval_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "tensorboard",
        run_name=f"qwen-coder-lora-r{lora_args.lora_rank}",
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print(f"\nTraining complete! Model saved to {output_dir}")
    print("\nNext steps:")
    print(f"1. Merge LoRA weights: python training/merge_lora.py --lora_path {output_dir}")
    print("2. Evaluate: python evaluation/run_evalplus.py")


if __name__ == "__main__":
    main()
