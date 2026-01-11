#!/bin/bash
# GRPO Training Launch Script
#
# Usage:
#   # Simple training (no OpenRLHF)
#   bash training/train_grpo_ray.sh --simple
#
#   # With OpenRLHF
#   bash training/train_grpo_ray.sh
#
#   # With Ray distributed
#   bash training/train_grpo_ray.sh --ray

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-configs/grpo_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen-coder-grpo}"

# Parse arguments
USE_RAY=false
USE_SIMPLE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --ray)
            USE_RAY=true
            shift
            ;;
        --simple)
            USE_SIMPLE=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ray] [--simple] [--config CONFIG] [--output OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

# Change to project directory
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if data exists
if [ ! -f "data/rl_prompts/mbpp_prompts.jsonl" ]; then
    echo "RL training data not found. Downloading..."
    python data/download_rl_dataset.py --dataset mbpp
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GRPO Training Configuration"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Mode: $([ "$USE_SIMPLE" = true ] && echo "Simple" || ([ "$USE_RAY" = true ] && echo "Ray Distributed" || echo "OpenRLHF"))"
echo "=========================================="

# Build training command
TRAIN_CMD="python training/train_grpo.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR"

if [ "$USE_SIMPLE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --simple"
elif [ "$USE_RAY" = true ]; then
    # Start Ray if not running
    if ! ray status 2>/dev/null; then
        echo "Starting Ray head node..."
        ray start --head --dashboard-host 0.0.0.0
        sleep 5
    fi
    TRAIN_CMD="$TRAIN_CMD --use_ray"
fi

# Run training
echo ""
echo "Running: $TRAIN_CMD"
echo ""

$TRAIN_CMD

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python evaluation/run_evalplus.py --model_path $OUTPUT_DIR/final"
echo "  2. Compare with SFT: python evaluation/compare_checkpoints.py \\"
echo "       --sft outputs/qwen-coder-lora/merged \\"
echo "       --rl $OUTPUT_DIR/final"
