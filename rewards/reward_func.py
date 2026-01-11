"""
Custom reward function for OpenRLHF GRPO training.

This module implements the reward function that evaluates generated code
by executing it against test cases.

Usage with OpenRLHF:
    --reward_pretrain ./rewards/reward_func.py
    --custom_reward_func reward_func
"""

import os
import re
import json
import sys
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards.code_executor import CodeExecutor, ExecutionResult


# Global executor instance
_executor = None


def get_executor():
    """Get or create executor instance."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor(timeout=10)
    return _executor


def extract_code_from_response(response: str) -> str:
    """Extract Python code from model response."""
    # Try to extract from markdown code block
    if "```python" in response:
        match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

    if "```" in response:
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Remove common prefixes/suffixes
    response = response.strip()

    # Remove explanation text before code
    lines = response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', '@')):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines)

    return response


def parse_test_cases(label: str) -> Dict[str, Any]:
    """Parse test cases from label string."""
    if not label:
        return {"test_cases": [], "type": "assert"}

    try:
        # Try JSON format first
        data = json.loads(label)
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            return {"test_cases": data, "type": "assert"}
    except json.JSONDecodeError:
        pass

    # Try eval for Python list format
    try:
        test_list = eval(label)
        if isinstance(test_list, list):
            return {"test_cases": test_list, "type": "assert"}
    except Exception:
        pass

    # Assume it's a single assert statement
    return {"test_cases": [label], "type": "assert"}


def compute_single_reward(
    response: str,
    label: str,
    executor: CodeExecutor,
) -> Dict[str, Any]:
    """Compute reward for a single response."""
    # Extract code from response
    code = extract_code_from_response(response)

    if not code.strip():
        return {
            "reward": -0.5,
            "score": 0.0,
            "status": "empty",
            "passed": 0,
            "total": 1
        }

    # Parse test cases
    test_info = parse_test_cases(label)
    test_type = test_info.get("type", "assert")

    if test_type == "stdin":
        # Stdin/stdout tests (competitive programming)
        inputs = test_info.get("inputs", [])
        outputs = test_info.get("outputs", [])

        if not inputs:
            return {
                "reward": 0.0,
                "score": 0.5,
                "status": "no_tests",
                "passed": 0,
                "total": 0
            }

        try:
            result = executor.execute_stdin_stdout(code, inputs, outputs)
        except Exception as e:
            return {
                "reward": -0.5,
                "score": 0.0,
                "status": "error",
                "error": str(e),
                "passed": 0,
                "total": len(inputs)
            }
    else:
        # Assert-style tests (MBPP)
        test_cases = test_info.get("test_cases", [])

        if not test_cases:
            return {
                "reward": 0.0,
                "score": 0.5,
                "status": "no_tests",
                "passed": 0,
                "total": 0
            }

        try:
            result = executor.execute_mbpp_style(
                code,
                test_cases,
                entry_point=test_info.get("entry_point"),
                setup_code=test_info.get("test_setup_code", "")
            )
        except Exception as e:
            return {
                "reward": -0.5,
                "score": 0.0,
                "status": "error",
                "error": str(e),
                "passed": 0,
                "total": len(test_cases)
            }

    # Compute reward based on test pass rate
    if result.num_total > 0:
        pass_rate = result.num_passed / result.num_total
    else:
        pass_rate = 0.0

    # Binary reward (DeepSeek style)
    if result.passed:
        reward = 1.0
        score = 1.0
    else:
        reward = 0.0
        score = pass_rate  # Partial score for logging

    return {
        "reward": reward,
        "score": score,
        "status": result.status,
        "passed": result.num_passed,
        "total": result.num_total,
        "execution_time": result.execution_time
    }


def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: Optional[List[str]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    OpenRLHF-compatible reward function.

    Args:
        queries: List of full sequences (prompt + response)
        prompts: List of original prompts
        labels: List of test cases (from dataset's label_key)

    Returns:
        Tensor of reward values
    """
    executor = get_executor()

    batch_size = len(queries)
    rewards = []

    # Extract responses from queries
    responses = []
    for query, prompt in zip(queries, prompts):
        # Response is everything after the prompt
        if prompt in query:
            response = query[len(prompt):].strip()
        else:
            # Fallback: look for assistant marker
            if "<|im_start|>assistant" in query:
                response = query.split("<|im_start|>assistant")[-1].strip()
                # Remove end token if present
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
            else:
                response = query
        responses.append(response)

    # Handle missing labels
    if labels is None:
        labels = [""] * batch_size

    # Process in parallel for efficiency
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(compute_single_reward, resp, label, executor): i
            for i, (resp, label) in enumerate(zip(responses, labels))
        }

        results = [None] * batch_size
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"reward": -0.5, "status": "error", "error": str(e)}

    # Extract rewards
    for result in results:
        rewards.append(result["reward"])

    return torch.tensor(rewards, dtype=torch.float32)


def get_reward_with_details(
    queries: List[str],
    prompts: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get rewards with detailed logging information.

    Returns:
        Dictionary with rewards tensor and extra logs
    """
    executor = get_executor()

    batch_size = len(queries)
    rewards = []
    scores = []
    statuses = []
    pass_counts = []

    # Extract responses
    responses = []
    for query, prompt in zip(queries, prompts):
        if prompt in query:
            response = query[len(prompt):].strip()
        else:
            if "<|im_start|>assistant" in query:
                response = query.split("<|im_start|>assistant")[-1].strip()
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
            else:
                response = query
        responses.append(response)

    if labels is None:
        labels = [""] * batch_size

    # Process in parallel
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(compute_single_reward, resp, label, executor): i
            for i, (resp, label) in enumerate(zip(responses, labels))
        }

        results = [None] * batch_size
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "reward": -0.5,
                    "score": 0.0,
                    "status": "error",
                    "passed": 0,
                    "total": 1
                }

    for result in results:
        rewards.append(result["reward"])
        scores.append(result.get("score", 0.0))
        statuses.append(result.get("status", "unknown"))
        pass_counts.append(result.get("passed", 0))

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "extra_logs": {
            "pass_rate": sum(scores) / len(scores) if scores else 0.0,
            "pass_count": sum(pass_counts),
            "total_tests": sum(r.get("total", 0) for r in results),
            "status_counts": {s: statuses.count(s) for s in set(statuses)}
        }
    }


# For testing
if __name__ == "__main__":
    print("Testing reward function...")

    # Test case 1: Correct code
    test_queries = [
        "<|im_start|>user\nWrite a function to add two numbers<|im_end|>\n<|im_start|>assistant\ndef add(a, b):\n    return a + b<|im_end|>",
    ]
    test_prompts = [
        "<|im_start|>user\nWrite a function to add two numbers<|im_end|>\n<|im_start|>assistant\n",
    ]
    test_labels = [
        '["assert add(1, 2) == 3", "assert add(0, 0) == 0"]',
    ]

    result = reward_func(test_queries, test_prompts, test_labels)
    print(f"Test 1 (correct code): reward = {result[0].item()}")

    # Test case 2: Incorrect code
    test_queries2 = [
        "<|im_start|>user\nWrite a function to add two numbers<|im_end|>\n<|im_start|>assistant\ndef add(a, b):\n    return a - b<|im_end|>",
    ]

    result2 = reward_func(test_queries2, test_prompts, test_labels)
    print(f"Test 2 (incorrect code): reward = {result2[0].item()}")

    # Test case 3: With details
    detailed = get_reward_with_details(test_queries + test_queries2, test_prompts * 2, test_labels * 2)
    print(f"Test 3 (detailed): rewards = {detailed['rewards'].tolist()}")
    print(f"  Extra logs: {detailed['extra_logs']}")

    print("\nAll tests completed!")
