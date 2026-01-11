"""
Compare Base and RL checkpoints on code generation benchmarks using vLLM.

Usage:
    python evaluation/compare_checkpoints.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --rl_model outputs/qwen-coder-grpo/final \
        --benchmarks humaneval mbpp
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from vllm import LLM, SamplingParams
from tqdm import tqdm


def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def extract_code(response: str) -> str:
    """Extract clean code from model response."""
    # Remove stop sequences
    stop_sequences = ["<|im_end|>", "\n\nHuman:", "\n\nUser:"]
    for stop in stop_sequences:
        if stop in response:
            response = response[:response.index(stop)]

    # Extract from markdown code blocks
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
) -> List[str]:
    """Generate completions for a batch of prompts."""

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    formatted_prompts = [format_prompt(p) for p in prompts]
    outputs = llm.generate(formatted_prompts, sampling_params)

    completions = []
    for output in outputs:
        text = output.outputs[0].text
        completions.append(extract_code(text))

    return completions


def run_humaneval(
    llm: LLM,
    output_dir: str,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> Dict:
    """Run HumanEval+ evaluation."""
    try:
        from evalplus.data import get_human_eval_plus, write_jsonl
        from evalplus.evaluate import evaluate
    except ImportError:
        print("Warning: evalplus not installed, skipping HumanEval")
        return {}

    print("\n" + "=" * 50)
    print("Running HumanEval+ Evaluation (vLLM)")
    print("=" * 50)

    problems = get_human_eval_plus()

    task_ids = list(problems.keys())
    prompts = [
        f"Complete the following Python function:\n\n{problems[tid]['prompt']}"
        for tid in task_ids
    ]

    print(f"Total problems: {len(prompts)}")

    # Batch generate
    all_completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        completions = batch_generate(llm, batch_prompts, max_new_tokens)
        all_completions.extend(completions)

    results = [
        {"task_id": tid, "completion": comp}
        for tid, comp in zip(task_ids, all_completions)
    ]

    # Save generations
    output_file = os.path.join(output_dir, "humaneval_generations.jsonl")
    write_jsonl(output_file, results)

    # Run evaluation
    print("\nEvaluating...")
    try:
        eval_results = evaluate(
            dataset="humaneval",
            samples=output_file,
            base_only=False,
            parallel=4,
        )
        return eval_results
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {}


def run_mbpp(
    llm: LLM,
    output_dir: str,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> Dict:
    """Run MBPP+ evaluation."""
    try:
        from evalplus.data import get_mbpp_plus, write_jsonl
        from evalplus.evaluate import evaluate
    except ImportError:
        print("Warning: evalplus not installed, skipping MBPP")
        return {}

    print("\n" + "=" * 50)
    print("Running MBPP+ Evaluation (vLLM)")
    print("=" * 50)

    problems = get_mbpp_plus()

    task_ids = list(problems.keys())
    prompts = [
        f"Write a Python function to solve this problem:\n{problems[tid]['prompt']}"
        for tid in task_ids
    ]

    print(f"Total problems: {len(prompts)}")

    # Batch generate
    all_completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        completions = batch_generate(llm, batch_prompts, max_new_tokens)
        all_completions.extend(completions)

    results = [
        {"task_id": tid, "completion": comp}
        for tid, comp in zip(task_ids, all_completions)
    ]

    output_file = os.path.join(output_dir, "mbpp_generations.jsonl")
    write_jsonl(output_file, results)

    print("\nEvaluating...")
    try:
        eval_results = evaluate(
            dataset="mbpp",
            samples=output_file,
            base_only=False,
            parallel=4,
        )
        return eval_results
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {}


def compare_models(
    base_path: str,
    rl_path: str,
    output_dir: str,
    benchmarks: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> Dict:
    """Run evaluation on both models and compare."""

    results = {
        "base": {"path": base_path, "humaneval": {}, "mbpp": {}},
        "rl": {"path": rl_path, "humaneval": {}, "mbpp": {}},
        "comparison": {},
        "timestamp": datetime.now().isoformat(),
    }

    for model_type, model_path in [("base", base_path), ("rl", rl_path)]:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model: {model_path}")
        print('='*60)

        # Check if model exists (for local paths)
        if not model_path.startswith(("Qwen/", "meta-llama/", "mistralai/")) and not os.path.exists(model_path):
            print(f"Warning: Model path not found: {model_path}")
            continue

        # Load model with vLLM
        print(f"Loading model with vLLM: {model_path}")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=2560,
        )
        print("Model loaded!")

        # Create output subdirectory
        model_output_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)

        # Run benchmarks
        if "humaneval" in benchmarks:
            humaneval_results = run_humaneval(
                llm, model_output_dir,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            results[model_type]["humaneval"] = humaneval_results

        if "mbpp" in benchmarks:
            mbpp_results = run_mbpp(
                llm, model_output_dir,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            results[model_type]["mbpp"] = mbpp_results

        # Free memory
        del llm
        import torch
        torch.cuda.empty_cache()

    # Compute comparison metrics
    for benchmark in benchmarks:
        base_results = results["base"].get(benchmark, {})
        rl_results = results["rl"].get(benchmark, {})

        if base_results and rl_results:
            base_score = base_results.get("pass@1", 0)
            rl_score = rl_results.get("pass@1", 0)

            # Handle case where scores might be percentages or decimals
            if isinstance(base_score, str):
                base_score = float(base_score.strip('%')) / 100
            if isinstance(rl_score, str):
                rl_score = float(rl_score.strip('%')) / 100

            improvement = rl_score - base_score
            relative_improvement = (improvement / max(base_score, 0.001)) * 100 if base_score else 0

            results["comparison"][benchmark] = {
                "base_pass@1": base_score,
                "rl_pass@1": rl_score,
                "improvement": improvement,
                "relative_improvement": relative_improvement
            }

    return results


def generate_comparison_report(results: Dict, output_dir: str) -> str:
    """Generate comparison report."""

    report = f"""
# Base vs RL Checkpoint Comparison Report

**Date**: {results.get('timestamp', 'N/A')}

## Models Evaluated

- **Base Model**: {results['base']['path']}
- **RL Model**: {results['rl']['path']}

## Results Summary

| Benchmark | Base pass@1 | RL pass@1 | Improvement | Relative % |
|-----------|-------------|-----------|-------------|------------|
"""

    for benchmark, comparison in results.get("comparison", {}).items():
        base_score = comparison['base_pass@1']
        rl_score = comparison['rl_pass@1']
        improvement = comparison['improvement']
        relative = comparison['relative_improvement']

        # Format as percentages
        if isinstance(base_score, float) and base_score <= 1:
            base_str = f"{base_score:.1%}"
            rl_str = f"{rl_score:.1%}"
            imp_str = f"{'+' if improvement >= 0 else ''}{improvement:.1%}"
        else:
            base_str = f"{base_score:.1f}%"
            rl_str = f"{rl_score:.1f}%"
            imp_str = f"{'+' if improvement >= 0 else ''}{improvement:.1f}%"

        report += f"| {benchmark} | {base_str} | {rl_str} | {imp_str} | {'+' if relative >= 0 else ''}{relative:.1f}% |\n"

    report += """
## Analysis

"""

    if results.get("comparison"):
        avg_improvement = sum(c["improvement"] for c in results["comparison"].values()) / len(results["comparison"])

        if avg_improvement > 0.05:
            report += "The RL training shows **significant improvement** over the base model.\n"
        elif avg_improvement > 0:
            report += "The RL training shows **modest improvement** over the base model.\n"
        else:
            report += "The RL training shows **no improvement** or slight regression compared to base.\n"

    # Save report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Also save as JSON
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {json_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Compare Base and RL checkpoints")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--rl_model", type=str, required=True, help="Path to RL checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/comparison", help="Output directory")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["humaneval", "mbpp"], help="Benchmarks to run")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run comparison
    results = compare_models(
        base_path=args.base_model,
        rl_path=args.rl_model,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # Generate report
    report_path = generate_comparison_report(results, args.output_dir)

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)

    # Print summary
    print("\nSummary:")
    for benchmark, comparison in results.get("comparison", {}).items():
        base = comparison['base_pass@1']
        rl = comparison['rl_pass@1']
        imp = comparison['improvement']

        if isinstance(base, float) and base <= 1:
            print(f"  {benchmark}: Base {base:.1%} -> RL {rl:.1%} ({'+' if imp >= 0 else ''}{imp:.1%})")
        else:
            print(f"  {benchmark}: Base {base:.1f}% -> RL {rl:.1f}% ({'+' if imp >= 0 else ''}{imp:.1f}%)")


if __name__ == "__main__":
    main()
