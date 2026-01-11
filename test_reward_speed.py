"""Test reward function execution speed."""
import time
import sys
sys.path.insert(0, '.')

from rewards.reward_func import reward_func

# Simple test case
responses = [
    "def add(a, b):\n    return a + b",
    "def add(a, b):\n    return a - b",  # Wrong
    "def add(a, b):\n    return a + b",
] * 10  # 30 samples

prompts = [""] * len(responses)
test_cases = ['["assert add(1, 2) == 3", "assert add(0, 0) == 0"]'] * len(responses)

print(f"Testing reward function with {len(responses)} samples...")
start = time.time()
rewards = reward_func(responses, prompts, test_cases)
elapsed = time.time() - start

print(f"Completed in {elapsed:.2f} seconds")
print(f"Speed: {len(responses)/elapsed:.2f} samples/second")
print(f"Estimated time for 384 samples: {384/(len(responses)/elapsed):.2f} seconds")
print(f"Rewards: {rewards[:5]}")
