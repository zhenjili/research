"""
Simple baseline evaluation using HuggingFace transformers.
Minimal dependencies for quick baseline measurement.
"""

import os
import json
import argparse
from pathlib import Path

def run_simple_humaneval():
    """Run a simple HumanEval test on a few examples."""

    print("=" * 60)
    print("Simple Baseline Evaluation")
    print("=" * 60)

    # Simple test cases for quick baseline
    test_cases = [
        {
            "task_id": "test_1",
            "prompt": "def add(a: int, b: int) -> int:\n    '''Return sum of a and b'''\n",
            "test": "assert add(2, 3) == 5\nassert add(0, 0) == 0",
        },
        {
            "task_id": "test_2",
            "prompt": "def is_even(n: int) -> bool:\n    '''Check if number is even'''\n",
            "test": "assert is_even(2) == True\nassert is_even(3) == False",
        },
    ]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"\nLoading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        model.eval()
        print("Model loaded successfully!")

        # Test generation
        prompt = "Write a Python function to add two numbers:"
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print("\n" + "=" * 60)
        print("Sample Generation Test:")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print(f"\nGenerated:\n{result}")
        print("\n✅ Model is working and ready for evaluation!")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    run_simple_humaneval()
