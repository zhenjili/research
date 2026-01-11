"""
Download and prepare training datasets for code generation fine-tuning.

Supported datasets:
- Evol-Instruct-Code-80k (WizardCoder)
- CodeAlpaca-20k
- OSS-Instruct (Magicoder)
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_evol_instruct_code(output_dir: str) -> str:
    """Download Evol-Instruct-Code-80k dataset."""
    print("Downloading Evol-Instruct-Code-80k...")

    # WizardLM's Evol-Instruct dataset for code
    dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train")

    # Filter for code-related instructions
    code_keywords = ['code', 'function', 'program', 'python', 'java', 'javascript',
                     'algorithm', 'implement', 'write', 'debug', 'fix', 'class',
                     'api', 'sql', 'html', 'css', 'script']

    def is_code_related(example):
        text = example.get('instruction', '').lower() + ' ' + example.get('output', '').lower()
        return any(kw in text for kw in code_keywords)

    code_dataset = dataset.filter(is_code_related)
    print(f"Filtered to {len(code_dataset)} code-related examples")

    # Convert to alpaca format
    alpaca_data = []
    for item in tqdm(code_dataset, desc="Converting"):
        alpaca_data.append({
            "instruction": item.get("instruction", ""),
            "input": "",
            "output": item.get("output", "")
        })

    output_path = os.path.join(output_dir, "evol_instruct_code.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(alpaca_data)} examples to {output_path}")
    return output_path


def download_code_alpaca(output_dir: str) -> str:
    """Download CodeAlpaca-20k dataset."""
    print("Downloading CodeAlpaca-20k...")

    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    alpaca_data = []
    for item in tqdm(dataset, desc="Converting"):
        alpaca_data.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        })

    output_path = os.path.join(output_dir, "code_alpaca_20k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(alpaca_data)} examples to {output_path}")
    return output_path


def download_oss_instruct(output_dir: str) -> str:
    """Download OSS-Instruct dataset (Magicoder)."""
    print("Downloading OSS-Instruct (Magicoder)...")

    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

    alpaca_data = []
    for item in tqdm(dataset, desc="Converting"):
        # Magicoder format: problem -> solution
        alpaca_data.append({
            "instruction": item.get("problem", ""),
            "input": "",
            "output": item.get("solution", "")
        })

    output_path = os.path.join(output_dir, "oss_instruct_75k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(alpaca_data)} examples to {output_path}")
    return output_path


def download_code_feedback(output_dir: str) -> str:
    """Download CodeFeedback dataset."""
    print("Downloading CodeFeedback...")

    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")

    alpaca_data = []
    for item in tqdm(dataset, desc="Converting"):
        alpaca_data.append({
            "instruction": item.get("query", ""),
            "input": "",
            "output": item.get("answer", "")
        })

    output_path = os.path.join(output_dir, "code_feedback.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(alpaca_data)} examples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download code instruction datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["evol_instruct", "code_alpaca", "oss_instruct", "code_feedback", "all"],
        default="oss_instruct",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded data"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    if args.dataset in ["evol_instruct", "all"]:
        path = download_evol_instruct_code(str(output_dir))
        downloaded.append(path)

    if args.dataset in ["code_alpaca", "all"]:
        path = download_code_alpaca(str(output_dir))
        downloaded.append(path)

    if args.dataset in ["oss_instruct", "all"]:
        path = download_oss_instruct(str(output_dir))
        downloaded.append(path)

    if args.dataset in ["code_feedback", "all"]:
        path = download_code_feedback(str(output_dir))
        downloaded.append(path)

    print("\n" + "=" * 50)
    print("Download complete!")
    print("Downloaded files:")
    for p in downloaded:
        print(f"  - {p}")
    print("\nNext: Run training with:")
    print("  python training/train_lora.py --data_path <path_to_json>")


if __name__ == "__main__":
    main()
