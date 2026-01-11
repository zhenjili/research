"""
Fast evaluation using vLLM for batch inference.
~5-10x faster than standard transformers generate.

Usage:
    python evaluation/run_evalplus_vllm.py --model_path Qwen/Qwen2.5-7B-Instruct --baseline
"""

import os
import json
import argparse
from datetime import datetime
from typing import List

from vllm import LLM, SamplingParams
from tqdm import tqdm

# EvalPlus imports
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from evalplus.evaluate import evaluate


def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def extract_code_from_response(response: str) -> str:
    """Extract clean code from model response."""
    # Remove common stop sequences
    stop_sequences = ["<|im_end|>", "\n\nHuman:", "\n\nUser:", "```\n\n"]
    for stop in stop_sequences:
        if stop in response:
            response = response[:response.index(stop)]

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


def batch_generate(
    llm: LLM,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> List[str]:
    """Generate completions for a batch of prompts."""

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    # Format all prompts
    formatted_prompts = [format_prompt(p) for p in prompts]

    # Batch generate
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract completions
    completions = []
    for output in outputs:
        text = output.outputs[0].text
        completions.append(extract_code_from_response(text))

    return completions


def run_humaneval(
    llm: LLM,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> dict:
    """Run HumanEval+ evaluation with batch inference."""

    print("\n" + "=" * 50)
    print("Running HumanEval+ Evaluation (vLLM)")
    print("=" * 50)

    problems = get_human_eval_plus()

    # Collect all prompts
    task_ids = list(problems.keys())
    prompts = [problems[tid]["prompt"] for tid in task_ids]

    print(f"Total problems: {len(prompts)}")
    print(f"Batch size: {batch_size}")

    # Batch generate
    all_completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        completions = batch_generate(llm, batch_prompts, max_new_tokens, temperature)
        all_completions.extend(completions)

    # Create results
    results = [
        {"task_id": tid, "completion": comp}
        for tid, comp in zip(task_ids, all_completions)
    ]

    # Save generations
    output_file = os.path.join(output_dir, "humaneval_generations.jsonl")
    write_jsonl(output_file, results)
    print(f"Saved generations to: {output_file}")

    # Run evaluation
    print("\nEvaluating generated code...")
    print(f"  - Samples file: {output_file}")
    print(f"  - Running {len(results)} test cases with 8 parallel workers")
    print("  - This may take a few minutes...")

    eval_results = evaluate(
        dataset="humaneval",
        samples=output_file,
        base_only=False,
        parallel=8,
        i_just_wanna_run=True,  # Force re-evaluation
    )

    print("Evaluation complete!")
    return eval_results


def run_mbpp(
    llm: LLM,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> dict:
    """Run MBPP+ evaluation with batch inference."""

    print("\n" + "=" * 50)
    print("Running MBPP+ Evaluation (vLLM)")
    print("=" * 50)

    problems = get_mbpp_plus()

    # Collect all prompts
    task_ids = list(problems.keys())
    prompts = [
        f"Write a Python function to solve this problem:\n{problems[tid]['prompt']}"
        for tid in task_ids
    ]

    print(f"Total problems: {len(prompts)}")
    print(f"Batch size: {batch_size}")

    # Batch generate
    all_completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        completions = batch_generate(llm, batch_prompts, max_new_tokens, temperature)
        all_completions.extend(completions)

    # Create results
    results = [
        {"task_id": tid, "completion": comp}
        for tid, comp in zip(task_ids, all_completions)
    ]

    # Save generations
    output_file = os.path.join(output_dir, "mbpp_generations.jsonl")
    write_jsonl(output_file, results)
    print(f"Saved generations to: {output_file}")

    # Run evaluation
    print("\nEvaluating generated code...")
    print(f"  - Samples file: {output_file}")
    print(f"  - Running {len(results)} test cases with 8 parallel workers")
    print("  - This may take a few minutes...")

    eval_results = evaluate(
        dataset="mbpp",
        samples=output_file,
        base_only=False,
        parallel=8,
        i_just_wanna_run=True,  # Force re-evaluation
    )

    print("Evaluation complete!")
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
# Code Generation Evaluation Report (vLLM)

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
    parser = argparse.ArgumentParser(description="Fast evaluation using vLLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model name",
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
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

    # Initialize vLLM
    print(f"Loading model with vLLM: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2560,  # prompt + generation
    )
    print("Model loaded!")

    # Run evaluations
    humaneval_results = {}
    mbpp_results = {}

    if "humaneval" in args.benchmarks:
        humaneval_results = run_humaneval(
            llm, output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    if "mbpp" in args.benchmarks:
        mbpp_results = run_mbpp(
            llm, output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
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
