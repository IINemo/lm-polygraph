#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate up two levels to get to the project root (lm-polygraph/)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change the working dir to the project root
cd "$PROJECT_ROOT" || { echo "Error: Could not change to project root"; exit 1; }

echo "Working from Project Root: $(pwd)"

# "answer" or "reasoning" or "both"
CONFIG_TYPE="both" 

MODEL="ugrip_llama_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
SAMPLE_SIZE=1000 # e.g. 1000
BATCH_SIZE=20   
# SOFIA: NOTE I DONT THINK THIS GPU MEMORY UTILIZATION THING WORKED WHEN I TESTED IT
# GPU_MEM_UTIL=0.60 

# Config file name based on config type
if [ "$CONFIG_TYPE" != "answer" ] && [ "$CONFIG_TYPE" != "reasoning" ] && [ "$CONFIG_TYPE" != "both" ]; then 
  echo "Error: CONFIG_TYPE must be 'answer', 'reasoning', or 'both'"
  exit 1 
fi

# output path
LOG_DIR="ugrip_logs/${CONFIG_TYPE}_$(date +%Y%m%d)"
LOG_FILE="${LOG_DIR}/run_$(date +%H%M%S)_batch${BATCH_SIZE}_sample${SAMPLE_SIZE}.log"

# Create output dir if doesn't exist yet
mkdir -p "$LOG_DIR"

# Assumes we're in lm-polygraph root directory
source .venv/bin/activate

# Group commands, redirect stderr to stdout, and pipe to tee
{
  echo "========================================="
  echo "LM-Polygraph Answer/Reasoning Testing"
  echo "Host: $(hostname)"
  echo "Run Type: $CONFIG_TYPE"
  echo "Model: $MODEL"
  echo "Dataset: $DATASET"
  echo "Sample Size: $SAMPLE_SIZE"
  echo "Batch Size: $BATCH_SIZE"
  echo "Output will be mirrored to: $LOG_FILE"
  echo "========================================="
  echo ""

  TOTAL_START_TIME=$SECONDS

  # --- Run Reasoning ---
  if [ "$CONFIG_TYPE" == "reasoning" ] || [ "$CONFIG_TYPE" == "both" ]; then
    echo "--- STARTING REASONING ANALYSIS ---"
    TASK_START_TIME=$SECONDS
    
    # Set the correct HYDRA_CONFIG for this block
    export HYDRA_CONFIG=$(pwd)/examples/configs/polygraph_eval_ugrip_segmentation_reasoning.yaml
    
    uv run --python 3.11 scripts/polygraph_eval \
      model=$MODEL \
      dataset=$DATASET \
      subsample_eval_dataset=$SAMPLE_SIZE \
      batch_size=$BATCH_SIZE \
      --config-name polygraph_eval_ugrip_segmentation_reasoning.yaml # Explicitly pass config name

    ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
    echo "--- FINISHED REASONING ANALYSIS ---"
    echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
    echo "-----------------------------------------"
  fi

  # --- Run Answer ---
  if [ "$CONFIG_TYPE" == "answer" ] || [ "$CONFIG_TYPE" == "both" ]; then
    echo "--- STARTING ANSWER ANALYSIS ---"
    TASK_START_TIME=$SECONDS

    # Set the correct HYDRA_CONFIG for this block
    export HYDRA_CONFIG=$(pwd)/examples/configs/polygraph_eval_ugrip_segmentation_answer.yaml

    uv run --python 3.11 scripts/polygraph_eval \
      model=$MODEL \
      dataset=$DATASET \
      subsample_eval_dataset=$SAMPLE_SIZE \
      batch_size=$BATCH_SIZE \
      --config-name polygraph_eval_ugrip_segmentation_answer.yaml # Explicitly pass config name

    ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
    echo "--- FINISHED ANSWER ANALYSIS ---"
    echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
    echo "-----------------------------------------"
  fi

  # --- Final Summary ---
  TOTAL_ELAPSED_TIME=$(($SECONDS - $TOTAL_START_TIME))
  echo ""
  echo "========================================================"
  echo "All requested tasks finished."
  echo "Total Time Elapsed: $(($TOTAL_ELAPSED_TIME / 3600)) hrs, $((($TOTAL_ELAPSED_TIME / 60) % 60)) min, $(($TOTAL_ELAPSED_TIME % 60)) sec"
  echo "========================================================"

} 2>&1 | tee "$LOG_FILE"