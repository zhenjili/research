"""
Test vLLM generation and reward computation.
Quick test without training loop.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, '.')

from vllm import LLM, SamplingParams
from rewards.reward_func import reward_func

def format_prompt(prompt: str) -> str:
    """Format prompt for Qwen chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

print("=" * 60)
print("Testing vLLM Generation + Reward Computation")
print("=" * 60)

# Load a few test prompts
print("\n1. Loading test prompts...")
prompts_data = []
with open("./data/rl_prompts/combined_rl_prompts.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 4:  # Just 4 prompts
            break
        prompts_data.append(json.loads(line))

print(f"   Loaded {len(prompts_data)} prompts")

# Initialize vLLM
print("\n2. Initializing vLLM...")
vllm_model = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.4,
    trust_remote_code=True,
    dtype="bfloat16",
)
print("   vLLM initialized!")

# Prepare prompts (4 prompts × 4 samples = 16 total)
n_samples = 4
formatted_prompts = []
test_cases_list = []

for item in prompts_data:
    formatted = format_prompt(item["prompt"])
    for _ in range(n_samples):
        formatted_prompts.append(formatted)
        test_cases_list.append(item["test_cases"])

print(f"\n3. Generating {len(formatted_prompts)} samples with vLLM...")
gen_start = time.time()

sampling_params = SamplingParams(
    max_tokens=512,
    temperature=1.0,
    top_p=0.95,
    stop=["<|im_end|>", "<|im_start|>"],
)

outputs = vllm_model.generate(formatted_prompts, sampling_params)

completions = [output.outputs[0].text for output in outputs]
gen_time = time.time() - gen_start

print(f"   Generated {len(completions)} samples in {gen_time:.2f}s ({len(completions)/gen_time:.1f} samples/s)")
print(f"\n   Sample output (first 200 chars):")
print(f"   {completions[0][:200]}...")

# Compute rewards
print(f"\n4. Computing rewards for {len(completions)} samples...")
reward_start = time.time()

rewards = reward_func(completions, [""] * len(completions), test_cases_list)

reward_time = time.time() - reward_start
mean_reward = rewards.mean().item()

print(f"   Rewards computed in {reward_time:.2f}s")
print(f"   Mean reward: {mean_reward:.3f}")
print(f"   Reward distribution: {rewards.tolist()}")

print("\n" + "=" * 60)
print("✅ Test completed successfully!")
print(f"Total time: {gen_time + reward_time:.2f}s (gen: {gen_time:.2f}s, reward: {reward_time:.2f}s)")
print("=" * 60)
