#!/bin/bash

# Change this to either "answer" or "reasoning"
CONFIG_TYPE="answer" # CHANGE THIS to "reasoning" for the other run

MODEL="ugrip_llama_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
SAMPLE_SIZE=2
BATCH_SIZE=20   
# SOFIA: NOTE I DONT THINK THIS GPU MEMORY UTILIZATION THING WORKED WHEN I TESTED IT
GPU_MEM_UTIL=0.60 

# Config file name based on config type
if [ "$CONFIG_TYPE" == "answer" ]; then
  CONFIG_FILE_NAME="polygraph_eval_ugrip_segmentation_answer.yaml"
elif [ "$CONFIG_TYPE" == "reasoning" ]; then
  CONFIG_FILE_NAME="polygraph_eval_ugrip_segmentation_reasoning.yaml"
else
  echo "Error: CONFIG_TYPE must be 'answer' or 'reasoning'"
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
    echo "Config File: $CONFIG_FILE_NAME"
    echo "Model: $MODEL"
    echo "Dataset: $DATASET"
    echo "Sample Size: $SAMPLE_SIZE"
    echo "Batch Size: $BATCH_SIZE"
    echo "GPU Memory Util: $GPU_MEM_UTIL"
    echo "Output will be mirrored to: $LOG_FILE"
    echo "========================================="
    echo ""

    HYDRA_CONFIG=`pwd`/examples/configs/$CONFIG_FILE_NAME \
      uv run --python 3.11 scripts/polygraph_eval  model=$MODEL \
      dataset=$DATASET   subsample_eval_dataset=$SAMPLE_SIZE   \ 
      model.load_model_args.gpu_memory_utilization=$GPU_MEM_UTIL  batch_size=$BATCH_SIZE 

    # Set HYDRA_CONFIG environment variable (used by polygraph_eval if needed)
    export HYDRA_CONFIG=`pwd`/examples/configs/$CONFIG_FILE_NAME

    # Run the evaluation script
    uv run --python 3.11 scripts/polygraph_eval \
      model=$MODEL \
      dataset=$DATASET \
      subsample_eval_dataset=$SAMPLE_SIZE \
      model.load_model_args.gpu_memory_utilization=$GPU_MEM_UTIL \
      batch_size=$BATCH_SIZE \
      --config-name $CONFIG_FILE_NAME # Pass config name explicitly

    echo ""
    echo "--- Finished LM-Polygraph Run ---"
    echo "========================================================"

} 2>&1 | tee "$LOG_FILE"

echo "Script finished. A complete log is saved to: $LOG_FILE"