#!/bin/bash

# Navigate to the project's root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || { echo "Error: Could not change to project root"; exit 1; }

# Define the default settings
DEFAULT_MODEL="ugrip_llama_instruct_vllm"
DEFAULT_DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
DEFAULT_SAMPLE_SIZE=-1
DEFAULT_CONFIG_TYPE="all"
DEFAULT_BATCH_SIZE=20

# User prompt function
ask_setting() {
  local prompt_text=$1
  local default_val=$2
  local user_val

  read -p "$prompt_text [$default_val]: " user_val
  # If user_val is empty just use default_val
  echo "${user_val:-$default_val}"
}

# Prompt user for settings
echo "Note: Hit Enter to use a default setting."
MODEL=$(ask_setting "Enter Model Name" "$DEFAULT_MODEL")
DATASET=$(ask_setting "Enter Dataset Path" "$DEFAULT_DATASET")
SAMPLE_SIZE=$(ask_setting "Enter Sample Size (-1 for all)" "$DEFAULT_SAMPLE_SIZE")

while true; do
  echo "Config options:"
  echo "  - answer    : Answer only"
  echo "  - reasoning : Reasoning only"
  echo "  - both      : Runs answer only, reasoning only separately"
  echo "  - full      : The entire output"
  echo "  - all       : DEFAULT SETTING. Runs everything (full, reasoning, and answer)"
  CONFIG_TYPE=$(ask_setting "Enter Config Type" "$DEFAULT_CONFIG_TYPE")
  
  # Validate input
  if [[ "$CONFIG_TYPE" == "answer" || "$CONFIG_TYPE" == "reasoning" || "$CONFIG_TYPE" == "both" || "$CONFIG_TYPE" == "full" || "$CONFIG_TYPE" == "all" ]]; then
    break
  else
    echo "Error: Invalid config type '$CONFIG_TYPE'. Please enter 'answer', 'reasoning', 'both', 'full', or 'all'."
  fi
done

BATCH_SIZE=$(ask_setting "Enter Batch Size" "$DEFAULT_BATCH_SIZE")

# Set up logging
LOG_DIR="ugrip_logs/${CONFIG_TYPE}_$(date +%Y%m%d)"
LOG_FILE="${LOG_DIR}/custom_run_$(date +%H%M%S)_batch${BATCH_SIZE}.log"
mkdir -p "$LOG_DIR"

# Activate the environment if it hasn't been already
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Warning: .venv/bin/activate not found. Assuming environment is already active."
fi

{
  echo ""
  echo "========================================="
  echo "       STARTING CUSTOM BENCHMARK         "
  echo "========================================="
  echo "Host: $(hostname)"
  echo "Model:       $MODEL"
  echo "Dataset:     $DATASET"
  echo "Run Type:    $CONFIG_TYPE"
  echo "Sample Size: $SAMPLE_SIZE"
  echo "Batch Size:  $BATCH_SIZE"
  echo "Log File:    $LOG_FILE"
  echo "========================================="
  echo ""

  TOTAL_START_TIME=$SECONDS

  # -Run full (entire answer)
  if [ "$CONFIG_TYPE" == "full" ] || [ "$CONFIG_TYPE" == "all" ]; then
    echo "--- STARTING FULL (ENTIRE ANSWER) ANALYSIS ---"
    TASK_START_TIME=$SECONDS
    
    # Set the correct HYDRA_CONFIG for the standard run
    export HYDRA_CONFIG=$(pwd)/examples/configs/polygraph_eval_ugrip.yaml
    
    uv run --python 3.11 scripts/polygraph_eval \
      model="$MODEL" \
      dataset="$DATASET" \
      subsample_eval_dataset="$SAMPLE_SIZE" \
      batch_size="$BATCH_SIZE" \
      --config-name polygraph_eval_ugrip.yaml

    ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
    echo "--- FINISHED FULL ANALYSIS ---"
    echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
    echo "-----------------------------------------"
  fi

  # Run reasoning segmentation
  if [ "$CONFIG_TYPE" == "reasoning" ] || [ "$CONFIG_TYPE" == "both" ] || [ "$CONFIG_TYPE" == "all" ]; then
    echo "--- STARTING REASONING ANALYSIS ---"
    TASK_START_TIME=$SECONDS
    
    # Set the correct HYDRA_CONFIG for this block
    export HYDRA_CONFIG=$(pwd)/examples/configs/polygraph_eval_ugrip_segmentation_reasoning.yaml
    
    uv run --python 3.11 scripts/polygraph_eval \
      model="$MODEL" \
      dataset="$DATASET" \
      subsample_eval_dataset="$SAMPLE_SIZE" \
      batch_size="$BATCH_SIZE" \
      --config-name polygraph_eval_ugrip_segmentation_reasoning.yaml

    ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
    echo "--- FINISHED REASONING ANALYSIS ---"
    echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
    echo "-----------------------------------------"
  fi

  # Run answer segmentation
  if [ "$CONFIG_TYPE" == "answer" ] || [ "$CONFIG_TYPE" == "both" ] || [ "$CONFIG_TYPE" == "all" ]; then
    echo "--- STARTING ANSWER ANALYSIS ---"
    TASK_START_TIME=$SECONDS

    # Set the correct HYDRA_CONFIG for this block
    export HYDRA_CONFIG=$(pwd)/examples/configs/polygraph_eval_ugrip_segmentation_answer.yaml

    uv run --python 3.11 scripts/polygraph_eval \
      model="$MODEL" \
      dataset="$DATASET" \
      subsample_eval_dataset="$SAMPLE_SIZE" \
      batch_size="$BATCH_SIZE" \
      --config-name polygraph_eval_ugrip_segmentation_answer.yaml

    ELAPSED_TIME=$(($SECONDS - $TASK_START_TIME))
    echo "--- FINISHED ANSWER ANALYSIS ---"
    echo "Time Elapsed: $(($ELAPSED_TIME / 60)) min, $(($ELAPSED_TIME % 60)) sec"
    echo "-----------------------------------------"
  fi

  # Final summary
  TOTAL_ELAPSED_TIME=$(($SECONDS - $TOTAL_START_TIME))
  echo ""
  echo "========================================================"
  echo "All requested tasks finished."
  echo "Total Time Elapsed: $(($TOTAL_ELAPSED_TIME / 3600)) hrs, $((($TOTAL_ELAPSED_TIME / 60) % 60)) min, $(($TOTAL_ELAPSED_TIME % 60)) sec"
  echo "========================================================"

} 2>&1 | tee "$LOG_FILE"