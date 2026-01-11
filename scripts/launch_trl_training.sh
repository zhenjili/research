#!/bin/bash
# Launch script for TRL GRPO Training on 8x A100 80GB
#
# This script manages two processes:
# 1. vLLM server on GPUs 0-1 for fast generation
# 2. DeepSpeed training on GPUs 2-7 with Accelerate
#
# Usage:
#   bash scripts/launch_trl_training.sh [config_file]
#
# Example:
#   bash scripts/launch_trl_training.sh configs/grpo_config.yaml

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${1:-configs/grpo_config.yaml}"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # Small model for fast iteration
VLLM_GPUS="0"  # Single GPU for vLLM (simpler, more stable)
TRAINING_GPUS="1,2,3,4,5,6,7"  # 7 GPUs for training
ACCELERATE_CONFIG="configs/accelerate_trl_8xa100.yaml"

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/trl_training_${TIMESTAMP}"
VLLM_LOG="${LOG_DIR}/vllm_server.log"
TRAINING_LOG="${LOG_DIR}/training.log"

mkdir -p "${LOG_DIR}"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}TRL GRPO Training Launch Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Config file:       ${GREEN}${CONFIG_FILE}${NC}"
echo -e "Model:             ${GREEN}${MODEL_NAME}${NC}"
echo -e "vLLM GPUs:         ${GREEN}${VLLM_GPUS}${NC}"
echo -e "Training GPUs:     ${GREEN}${TRAINING_GPUS}${NC}"
echo -e "Accelerate config: ${GREEN}${ACCELERATE_CONFIG}${NC}"
echo -e "Log directory:     ${GREEN}${LOG_DIR}${NC}"
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Check if accelerate config exists
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo -e "${RED}Error: Accelerate config not found: ${ACCELERATE_CONFIG}${NC}"
    exit 1
fi

# Check if TRL is installed
if ! python -c "import trl" 2>/dev/null; then
    echo -e "${RED}Error: TRL is not installed. Install with: pip install trl[vllm]${NC}"
    exit 1
fi

# Check GPU availability
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ ${NUM_GPUS} -lt 8 ]; then
    echo -e "${YELLOW}Warning: Found ${NUM_GPUS} GPUs, expected 8. Proceeding anyway...${NC}"
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Step 1: Starting vLLM Server${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "GPUs: ${GREEN}${VLLM_GPUS}${NC}"
echo -e "Log:  ${GREEN}${VLLM_LOG}${NC}"
echo ""

# Start vLLM server in background
# Note: trl vllm-serve uses limited arguments, use single GPU for simplicity
CUDA_VISIBLE_DEVICES=${VLLM_GPUS} trl vllm-serve \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo -e "${GREEN}✓ vLLM server started (PID: ${VLLM_PID})${NC}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up processes...${NC}"
    if ps -p ${VLLM_PID} > /dev/null 2>&1; then
        echo -e "Stopping vLLM server (PID: ${VLLM_PID})"
        kill ${VLLM_PID} 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Wait for vLLM server to start
echo -e "${YELLOW}Waiting for vLLM server to initialize...${NC}"
MAX_WAIT=120  # seconds
WAIT_TIME=0
while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
    if grep -q "Uvicorn running" "${VLLM_LOG}" 2>/dev/null; then
        echo -e "${GREEN}✓ vLLM server is ready!${NC}"
        break
    fi
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    echo -n "."
done
echo ""

if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
    echo -e "${RED}Error: vLLM server failed to start within ${MAX_WAIT} seconds${NC}"
    echo -e "${YELLOW}Check vLLM logs: ${VLLM_LOG}${NC}"
    exit 1
fi

# Give it a bit more time to be fully ready
sleep 5

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Step 2: Starting Training${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "GPUs: ${GREEN}${TRAINING_GPUS}${NC}"
echo -e "Log:  ${GREEN}${TRAINING_LOG}${NC}"
echo ""

# Start training
CUDA_VISIBLE_DEVICES=${TRAINING_GPUS} accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    training/train_grpo_trl_enhanced.py \
    --config "${CONFIG_FILE}" \
    2>&1 | tee "${TRAINING_LOG}"

TRAINING_EXIT_CODE=$?

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Training Complete${NC}"
echo -e "${BLUE}================================================${NC}"

if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ Training finished successfully!${NC}"
else
    echo -e "${RED}✗ Training failed with exit code: ${TRAINING_EXIT_CODE}${NC}"
fi

echo ""
echo -e "Logs saved to: ${GREEN}${LOG_DIR}${NC}"
echo -e "  - vLLM server: ${VLLM_LOG}"
echo -e "  - Training:    ${TRAINING_LOG}"

exit ${TRAINING_EXIT_CODE}
