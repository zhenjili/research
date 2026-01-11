"""
Merge LoRA weights with base model.

Usage:
    python training/merge_lora.py --lora_path outputs/qwen-coder-lora
    python training/merge_lora.py --lora_path outputs/qwen-coder-lora --output_dir outputs/merged
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_name: str = None,
):
    """Merge LoRA weights with base model and save."""

    print(f"Loading base model: {base_model_path}")
    print(f"Loading LoRA weights: {lora_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    # Merge weights
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_dir)

    print("Merge complete!")

    # Optional: Push to HuggingFace Hub
    if push_to_hub and hub_repo_name:
        print(f"Pushing to HuggingFace Hub: {hub_repo_name}")
        model.push_to_hub(hub_repo_name, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_name)
        print("Push complete!")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: lora_path/merged)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo_name",
        type=str,
        default=None,
        help="HuggingFace Hub repository name",
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.lora_path, "merged")

    # Merge
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_repo_name=args.hub_repo_name,
    )

    print("\nNext steps:")
    print(f"1. Test inference: python inference/chat.py --model_path {args.output_dir}")
    print(f"2. Evaluate: python evaluation/run_evalplus.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()
