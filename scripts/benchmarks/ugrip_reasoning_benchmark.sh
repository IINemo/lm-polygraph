#!/bin/bash

MODEL="ugrip_gemma_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/medmcqa-reasoning-new"
SAMPLE_SIZE=1000
BATCH_SIZE=64
# SOFIA: NOTE I DONT THINK THIS GPU MEMORY UTILIZATION THING WORKED WHEN I TESTED IT
# GPU_MEM_UTIL=0.60 

# output path
LOG_DIR="ugrip_logs/$(date +%Y%m%d)"
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
  echo "Model: $MODEL"
  echo "Dataset: $DATASET"
  echo "Sample Size: $SAMPLE_SIZE"
  echo "Batch Size: $BATCH_SIZE"
  echo "Output will be mirrored to: $LOG_FILE"
  echo "========================================="
  echo ""

  TOTAL_START_TIME=$SECONDS

  echo "--- STARTING REASONING ANALYSIS ---"
  TASK_START_TIME=$SECONDS
    
  # Set the correct HYDRA_CONFIG for this block
  export HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_ugrip.yaml
    
  uv run --python 3.11 scripts/polygraph_eval \
    model=$MODEL \
    dataset=$DATASET \
    subsample_eval_dataset=$SAMPLE_SIZE \
    batch_size=$BATCH_SIZE \
    --config-name polygraph_eval_ugrip.yaml # Explicitly pass config name

  ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
  echo "--- FINISHED REASONING ANALYSIS ---"
  echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
  echo "-----------------------------------------"

  # --- Final Summary ---
  TOTAL_ELAPSED_TIME=$(($SECONDS - $TOTAL_START_TIME))
  echo ""
  echo "========================================================"
  echo "All requested tasks finished."
  echo "Total Time Elapsed: $(($TOTAL_ELAPSED_TIME / 3600)) hrs, $((($TOTAL_ELAPSED_TIME / 60) % 60)) min, $(($TOTAL_ELAPSED_TIME % 60)) sec"
  echo "========================================================"

} 2>&1 | tee "$LOG_FILE"