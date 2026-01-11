"""
Interactive chat interface for testing the fine-tuned code model.

Usage:
    # Test merged model
    python inference/chat.py --model_path outputs/qwen-coder-lora/merged

    # Test with LoRA adapter
    python inference/chat.py --model_path Qwen/Qwen2.5-Coder-7B-Instruct --lora_path outputs/qwen-coder-lora

    # Single prompt mode
    python inference/chat.py --model_path outputs/merged --prompt "Write a Python function to sort a list"
"""

import os
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def load_model(
    model_path: str,
    lora_path: Optional[str] = None,
):
    """Load model with optional LoRA adapter."""

    console.print(f"[bold blue]Loading model:[/] {model_path}")

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

    if lora_path:
        console.print(f"[bold blue]Loading LoRA:[/] {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    console.print("[bold green]Model loaded successfully![/]\n")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = True,
) -> str:
    """Generate response for a given prompt."""

    # Format prompt for Qwen chat template
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    # Decode full response
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Clean up response
    stop_sequences = ["<|im_end|>", "\n\nHuman:", "\n\nUser:"]
    for stop in stop_sequences:
        if stop in generated:
            generated = generated[:generated.index(stop)]

    return generated.strip()


def format_code_response(response: str) -> None:
    """Format and display code response with syntax highlighting."""

    # Check if response contains code blocks
    if "```python" in response:
        parts = response.split("```python")
        for i, part in enumerate(parts):
            if i == 0:
                if part.strip():
                    console.print(part.strip())
            else:
                if "```" in part:
                    code, rest = part.split("```", 1)
                    console.print(Syntax(code.strip(), "python", theme="monokai", line_numbers=True))
                    if rest.strip():
                        console.print(rest.strip())
                else:
                    console.print(Syntax(part.strip(), "python", theme="monokai", line_numbers=True))
    elif "```" in response:
        parts = response.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    console.print(part.strip())
            else:
                console.print(Syntax(part.strip(), "python", theme="monokai", line_numbers=True))
    else:
        console.print(response)


def interactive_chat(
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
):
    """Run interactive chat session."""

    console.print(Panel.fit(
        "[bold]Code Assistant Chat[/]\n"
        "Type your coding questions or requests.\n"
        "Commands: [cyan]/quit[/] to exit, [cyan]/clear[/] to clear history",
        title="Welcome",
        border_style="green",
    ))

    while True:
        try:
            console.print()
            user_input = console.input("[bold cyan]You:[/] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit", "/q"]:
                console.print("[yellow]Goodbye![/]")
                break

            if user_input.lower() == "/clear":
                console.clear()
                continue

            console.print("\n[bold green]Assistant:[/]")

            response = generate_response(
                model, tokenizer, user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=True,
            )

            # Response is already streamed, just add newline
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/]")
            continue


def single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
):
    """Generate response for a single prompt."""

    console.print(Panel(prompt, title="[bold]Prompt[/]", border_style="blue"))
    console.print()

    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stream=False,
    )

    console.print(Panel.fit(
        response,
        title="[bold]Response[/]",
        border_style="green",
    ))

    return response


def main():
    parser = argparse.ArgumentParser(description="Chat with fine-tuned code model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/qwen-coder-lora/merged",
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt mode (non-interactive)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        lora_path=args.lora_path,
    )

    if args.prompt:
        # Single prompt mode
        single_prompt(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        # Interactive mode
        interactive_chat(
            model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
