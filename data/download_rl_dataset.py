"""
Download and prepare RL training datasets for code generation.

Supported datasets:
- MBPP (Mostly Basic Python Problems) - with test cases
- CodeContests (DeepMind) - competitive programming
- HumanEval - for validation

Usage:
    python data/download_rl_dataset.py --dataset mbpp
    python data/download_rl_dataset.py --dataset codecontests
    python data/download_rl_dataset.py --dataset all
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm


def extract_function_name(code: str) -> str:
    """Extract function name from Python code."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else "solution"


def format_mbpp_prompt(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Format MBPP problem for RL training."""
    task_id = problem["task_id"]
    description = problem["text"]
    test_list = problem["test_list"]
    test_setup_code = problem.get("test_setup_code", "")
    code = problem.get("code", "")

    # Create prompt in Qwen chat format
    prompt = f"""Write a Python function to solve the following problem:

{description}

Your function should pass the following test cases:
{test_list[0] if test_list else ""}

Provide only the function implementation without any explanation."""

    return {
        "task_id": f"mbpp_{task_id}",
        "prompt": prompt,
        "test_cases": json.dumps(test_list),
        "test_setup_code": test_setup_code,
        "canonical_solution": code,
        "entry_point": extract_function_name(code),
        "difficulty": "easy",
        "source": "mbpp"
    }


def format_codecontests_prompt(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Format CodeContests problem for RL training."""
    name = problem.get("name", "unknown")
    description = problem.get("description", "")

    # Get public test cases
    public_tests = problem.get("public_tests", {})
    input_tests = public_tests.get("input", [])
    output_tests = public_tests.get("output", [])

    # Get private test cases for actual verification
    private_tests = problem.get("private_tests", {})
    private_inputs = private_tests.get("input", [])
    private_outputs = private_tests.get("output", [])

    # Create prompt showing only public tests
    example_io = ""
    for i, (inp, out) in enumerate(zip(input_tests[:2], output_tests[:2])):
        example_io += f"\nExample {i+1}:\nInput:\n{inp}\nOutput:\n{out}\n"

    prompt = f"""Solve the following competitive programming problem:

{description}

{example_io}

Write a complete Python solution that reads from stdin and writes to stdout."""

    # Combine all test cases for verification
    all_inputs = input_tests + private_inputs
    all_outputs = output_tests + private_outputs

    # Format as stdin/stdout test cases
    test_cases = json.dumps({
        "type": "stdin",
        "inputs": all_inputs,
        "outputs": all_outputs
    })

    return {
        "task_id": f"codecontests_{name}",
        "prompt": prompt,
        "test_cases": test_cases,
        "difficulty": str(problem.get("difficulty", "unknown")),
        "source": "codecontests",
        "time_limit": problem.get("time_limit", {}).get("seconds", 2) if isinstance(problem.get("time_limit"), dict) else 2
    }


def download_mbpp(output_dir: str, split: str = "test") -> str:
    """Download and process MBPP dataset."""
    print("Downloading MBPP dataset...")

    # Load from HuggingFace
    dataset = load_dataset("google-research-datasets/mbpp", split=split)

    processed = []
    for problem in tqdm(dataset, desc="Processing MBPP"):
        formatted = format_mbpp_prompt(problem)
        processed.append(formatted)

    # Save as JSONL for streaming
    output_path = os.path.join(output_dir, "mbpp_prompts.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} MBPP problems to {output_path}")
    return output_path


def download_mbpp_train(output_dir: str) -> str:
    """Download MBPP training split for RL training."""
    print("Downloading MBPP training split...")

    # Load train split
    dataset = load_dataset("google-research-datasets/mbpp", split="train")

    processed = []
    for problem in tqdm(dataset, desc="Processing MBPP train"):
        formatted = format_mbpp_prompt(problem)
        processed.append(formatted)

    output_path = os.path.join(output_dir, "mbpp_train_prompts.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} MBPP training problems to {output_path}")
    return output_path


def download_codecontests(output_dir: str, max_problems: int = 2000) -> str:
    """Download and process CodeContests dataset."""
    print("Downloading CodeContests dataset...")

    try:
        # Load from HuggingFace (this is a large dataset)
        dataset = load_dataset("deepmind/code_contests", split="train")
    except Exception as e:
        print(f"Warning: Could not download CodeContests: {e}")
        print("Skipping CodeContests dataset.")
        return None

    processed = []
    for problem in tqdm(dataset, desc="Processing CodeContests"):
        if len(processed) >= max_problems:
            break

        # Check if problem has test cases
        public_tests = problem.get("public_tests", {})
        private_tests = problem.get("private_tests", {})

        inputs = public_tests.get("input", []) + private_tests.get("input", [])
        outputs = public_tests.get("output", []) + private_tests.get("output", [])

        if inputs and outputs:  # Must have test cases
            formatted = format_codecontests_prompt(problem)
            processed.append(formatted)

    # Save as JSONL
    output_path = os.path.join(output_dir, "codecontests_prompts.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} CodeContests problems to {output_path}")
    return output_path


def download_apps(output_dir: str, max_problems: int = 5000) -> str:
    """Download and process APPS dataset."""
    print("Downloading APPS dataset...")

    try:
        dataset = load_dataset("codeparrot/apps", split="train")
    except Exception as e:
        print(f"Warning: Could not download APPS: {e}")
        return None

    processed = []
    for problem in tqdm(dataset, desc="Processing APPS"):
        if len(processed) >= max_problems:
            break

        question = problem.get("question", "")
        input_output = problem.get("input_output", "")

        if not question or not input_output:
            continue

        try:
            io_data = json.loads(input_output)
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])

            if not inputs or not outputs:
                continue
        except:
            continue

        # Create prompt
        prompt = f"""Solve the following programming problem:

{question}

Write a complete Python solution that reads from stdin and writes to stdout."""

        processed.append({
            "task_id": f"apps_{len(processed)}",
            "prompt": prompt,
            "test_cases": json.dumps({
                "type": "stdin",
                "inputs": inputs[:5],  # Limit test cases
                "outputs": outputs[:5]
            }),
            "difficulty": problem.get("difficulty", "unknown"),
            "source": "apps"
        })

    output_path = os.path.join(output_dir, "apps_prompts.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} APPS problems to {output_path}")
    return output_path


def download_humaneval_for_validation(output_dir: str) -> str:
    """Download HumanEval for validation during RL training."""
    print("Downloading HumanEval for validation...")

    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    except ImportError:
        print("Warning: evalplus not installed, skipping HumanEval")
        return None

    processed = []
    for task_id, problem in problems.items():
        # Create test cases from the test function
        test_code = problem.get("test", "")

        processed.append({
            "task_id": task_id,
            "prompt": problem["prompt"],
            "entry_point": problem["entry_point"],
            "test_cases": json.dumps({
                "type": "humaneval",
                "test_code": test_code,
                "entry_point": problem["entry_point"]
            }),
            "canonical_solution": problem.get("canonical_solution", ""),
            "source": "humaneval"
        })

    output_path = os.path.join(output_dir, "humaneval_validation.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} HumanEval problems to {output_path}")
    return output_path


def create_combined_dataset(output_dir: str) -> str:
    """Create combined RL training dataset."""
    files_to_combine = [
        "mbpp_prompts.jsonl",
        "mbpp_train_prompts.jsonl",
        "codecontests_prompts.jsonl",
        "apps_prompts.jsonl",
    ]

    combined = []

    for filename in files_to_combine:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        combined.append(json.loads(line))
            print(f"  Loaded {filename}")

    output_path = os.path.join(output_dir, "combined_rl_prompts.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Created combined dataset with {len(combined)} problems")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download RL training datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mbpp", "codecontests", "apps", "humaneval", "all"],
        default="mbpp",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/rl_prompts",
        help="Output directory"
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=2000,
        help="Maximum problems for CodeContests"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    downloaded = []

    if args.dataset in ["mbpp", "all"]:
        path = download_mbpp(args.output_dir)
        downloaded.append(path)
        path = download_mbpp_train(args.output_dir)
        downloaded.append(path)

    if args.dataset in ["codecontests", "all"]:
        path = download_codecontests(args.output_dir, args.max_problems)
        if path:
            downloaded.append(path)

    if args.dataset in ["apps", "all"]:
        path = download_apps(args.output_dir, args.max_problems)
        if path:
            downloaded.append(path)

    if args.dataset in ["humaneval", "all"]:
        path = download_humaneval_for_validation(args.output_dir)
        if path:
            downloaded.append(path)

    if args.dataset == "all":
        path = create_combined_dataset(args.output_dir)
        downloaded.append(path)

    print("\n" + "=" * 50)
    print("Download complete!")
    print("Downloaded files:")
    for p in downloaded:
        if p:
            print(f"  - {p}")
    print("\nNext: Start GRPO training with:")
    print("  python training/train_grpo.py --config configs/grpo_config.yaml")


if __name__ == "__main__":
    main()
