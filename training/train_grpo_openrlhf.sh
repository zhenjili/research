#!/bin/bash
# GRPO Training Script using OpenRLHF
# Usage: bash training/train_grpo_openrlhf.sh

set -e

# Configuration
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="./data/rl_prompts/combined_rl_prompts.jsonl"
OUTPUT_DIR="./outputs/qwen-1.5b-grpo"
REWARD_FUNC="./rewards/reward_func.py"

# Training hyperparameters (optimized for A10 24GB)
NUM_EPISODES=2000
ROLLOUT_BATCH_SIZE=8
MICRO_ROLLOUT_BATCH_SIZE=4
MICRO_TRAIN_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=4
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=512
LEARNING_RATE=5e-7

# GRPO specific
ADVANTAGE_ESTIMATOR="group_norm"  # GRPO uses group normalization
KL_COEF=0.001

echo "=============================================="
echo "GRPO Training with OpenRLHF"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Episodes: $NUM_EPISODES"
echo "=============================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate virtual environment
source .venv/bin/activate

# Run training with Ray
python -m openrlhf.cli.train_ppo_ray \
    --pretrain $MODEL_PATH \
    --reward_pretrain $MODEL_PATH \
    --save_path $OUTPUT_DIR \
    --prompt_data $DATASET_PATH \
    --input_key "prompt" \
    --num_episodes $NUM_EPISODES \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --micro_rollout_batch_size $MICRO_ROLLOUT_BATCH_SIZE \
    --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --actor_learning_rate $LEARNING_RATE \
    --critic_learning_rate $LEARNING_RATE \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef $KL_COEF \
    --normalize_reward \
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --save_steps 500 \
    --logging_steps 10 \
    --max_epochs 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --colocate_all_models \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.4 \
    --remote_rm_url $REWARD_FUNC

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python evaluation/run_evalplus_vllm.py --model_path $OUTPUT_DIR"
echo "  2. Compare: python evaluation/compare_checkpoints.py --base_model $MODEL_PATH --rl_model $OUTPUT_DIR"
