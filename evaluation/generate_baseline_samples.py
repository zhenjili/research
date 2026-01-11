"""
Generate baseline samples for EvalPlus evaluation.
Creates a JSONL file with model predictions for HumanEval problems.
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import evalplus data
from evalplus.data import get_human_eval_plus, write_jsonl


def generate_samples(
    model_name: str,
    output_file: str,
    max_new_tokens: int = 512,
    num_samples: int = 1,
):
    """Generate code samples for HumanEval+ problems."""

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print("Model loaded successfully!")

    # Get HumanEval+ problems
    problems = get_human_eval_plus()
    print(f"Total problems: {len(problems)}")

    results = []

    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]

        # Format for Qwen
        formatted_prompt = f"<|im_start|>user\nComplete the following Python function:\n\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

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
                do_sample=False,  # Greedy for baseline
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Clean up - remove explanations after code
        if "<|im_end|>" in generated:
            generated = generated[:generated.index("<|im_end|>")]

        # Extract code from markdown if present
        if "```python" in generated:
            start = generated.index("```python") + len("```python")
            if "```" in generated[start:]:
                end = generated.index("```", start)
                generated = generated[start:end]
        elif "```" in generated:
            start = generated.index("```") + 3
            if "```" in generated[start:]:
                end = generated.index("```", start)
                generated = generated[start:end]

        # Combine prompt + completion
        completion = prompt + generated.strip()

        results.append({
            "task_id": task_id,
            "completion": completion,
        })

    # Save results
    write_jsonl(output_file, results)
    print(f"\nSaved {len(results)} samples to: {output_file}")
    print(f"\nNext step: Evaluate with:")
    print(f"  evalplus.evaluate --dataset humaneval --samples {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_file", default="./evaluation_results/baseline_samples.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    generate_samples(
        model_name=args.model_name,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
    )
