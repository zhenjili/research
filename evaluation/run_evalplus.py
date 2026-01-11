"""
Evaluate code generation models using EvalPlus (HumanEval+ and MBPP+).

Usage:
    # Evaluate merged model
    python evaluation/run_evalplus.py --model_path outputs/qwen-coder-lora/merged

    # Evaluate base model for comparison
    python evaluation/run_evalplus.py --model_path Qwen/Qwen2.5-Coder-7B-Instruct --baseline

    # Evaluate with LoRA adapter (without merging)
    python evaluation/run_evalplus.py --model_path Qwen/Qwen2.5-Coder-7B-Instruct --lora_path outputs/qwen-coder-lora
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# EvalPlus imports
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from evalplus.evaluate import evaluate


def load_model(
    model_path: str,
    lora_path: Optional[str] = None,
    device: str = "cuda",
):
    """Load model with optional LoRA adapter."""

    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter if specified
    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> str:
    """Generate code completion for a given prompt."""

    # Format prompt for Qwen
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Extract code (stop at common end markers)
    stop_sequences = ["<|im_end|>", "\n\nHuman:", "\n\nUser:", "```\n\n"]
    for stop in stop_sequences:
        if stop in generated:
            generated = generated[:generated.index(stop)]

    return generated.strip()


def extract_code_from_response(response: str, entry_point: str) -> str:
    """Extract clean code from model response."""

    # If response contains markdown code block, extract it
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start) if "```" in response[start:] else len(response)
        response = response[start:end]
    elif "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start) if "```" in response[start:] else len(response)
        response = response[start:end]

    return response.strip()


def run_humaneval(
    model,
    tokenizer,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Run HumanEval+ evaluation."""

    print("\n" + "=" * 50)
    print("Running HumanEval+ Evaluation")
    print("=" * 50)

    # Get HumanEval+ problems
    problems = get_human_eval_plus()

    results = []

    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        # Generate completion
        completion = generate_code(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )

        # Clean up completion
        completion = extract_code_from_response(completion, entry_point)

        results.append({
            "task_id": task_id,
            "completion": completion,
        })

    # Save generations
    output_file = os.path.join(output_dir, "humaneval_generations.jsonl")
    write_jsonl(output_file, results)
    print(f"Saved generations to: {output_file}")

    # Run evaluation
    print("\nEvaluating...")
    eval_results = evaluate(
        dataset="humaneval",
        samples=output_file,
        base_only=False,  # Use EvalPlus (stricter tests)
        parallel=8,
    )

    return eval_results


def run_mbpp(
    model,
    tokenizer,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Run MBPP+ evaluation."""

    print("\n" + "=" * 50)
    print("Running MBPP+ Evaluation")
    print("=" * 50)

    # Get MBPP+ problems
    problems = get_mbpp_plus()

    results = []

    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        # MBPP format: prompt contains task description
        prompt = f"Write a Python function to solve this problem:\n{problem['prompt']}"
        entry_point = problem["entry_point"]

        # Generate completion
        completion = generate_code(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )

        # Clean up completion
        completion = extract_code_from_response(completion, entry_point)

        results.append({
            "task_id": task_id,
            "completion": completion,
        })

    # Save generations
    output_file = os.path.join(output_dir, "mbpp_generations.jsonl")
    write_jsonl(output_file, results)
    print(f"Saved generations to: {output_file}")

    # Run evaluation
    print("\nEvaluating...")
    eval_results = evaluate(
        dataset="mbpp",
        samples=output_file,
        base_only=False,
        parallel=8,
    )

    return eval_results


def generate_report(
    humaneval_results: dict,
    mbpp_results: dict,
    output_dir: str,
    model_name: str,
) -> str:
    """Generate evaluation report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
# Code Generation Evaluation Report

**Model**: {model_name}
**Date**: {timestamp}

## Results Summary

### HumanEval+
- **pass@1 (base)**: {humaneval_results.get('pass@1', 'N/A'):.2%}
- **pass@1 (plus)**: {humaneval_results.get('pass@1+', 'N/A'):.2%}

### MBPP+
- **pass@1 (base)**: {mbpp_results.get('pass@1', 'N/A'):.2%}
- **pass@1 (plus)**: {mbpp_results.get('pass@1+', 'N/A'):.2%}

## Detailed Results

See generated JSONL files for individual problem results.
"""

    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Also save as JSON
    results_json = {
        "model": model_name,
        "timestamp": timestamp,
        "humaneval": humaneval_results,
        "mbpp": mbpp_results,
    }

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on EvalPlus benchmarks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Mark this as baseline evaluation",
    )

    args = parser.parse_args()

    # Create output directory
    model_name = args.model_path.split("/")[-1]
    if args.baseline:
        model_name += "_baseline"
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        lora_path=args.lora_path,
    )

    # Run evaluations
    humaneval_results = {}
    mbpp_results = {}

    if "humaneval" in args.benchmarks:
        humaneval_results = run_humaneval(
            model, tokenizer, output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    if "mbpp" in args.benchmarks:
        mbpp_results = run_mbpp(
            model, tokenizer, output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Generate report
    report_path = generate_report(
        humaneval_results,
        mbpp_results,
        output_dir,
        model_name,
    )

    print("\n" + "=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
