#!/bin/bash
#SBATCH --job-name=ugrip_segmentation             # Job name
#SBATCH --output=slurm_logs/%A/output_%A_%a.txt   # Standard output and error log
#SBATCH --error=slurm_logs/%A/error_%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=10          # Number of CPU cores
#SBATCH --gres=gpu:2                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs
#SBATCH --array=0-2 # job array with 3 tasks 0 1 2

mkdir -p slurm_logs/$SLURM_JOB_ID

# Experiment settings
CONFIG_FILES=(
  "polygraph_eval_ugrip.yaml"                        # full answer
  "polygraph_eval_ugrip_segmentation_reasoning.yaml" # reasoning only
  "polygraph_eval_ugrip_segmentation_answer.yaml"    # answer only
)
MODEL="ugrip_gemma_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
SAMPLE_SIZE=10

CONFIG_FILE_NAME=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

# experiment info
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config file: $CONFIG_FILE_NAME"
echo "Model: $MODEL, Dataset: $DATASET, Sample Size: $SAMPLE_SIZE"

# Run experiments
scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=$CONFIG_FILE_NAME \
  model=$MODEL \
  dataset=$DATASET \
  subsample_eval_dataset=$SAMPLE_SIZE

echo "Task $SLURM_ARRAY_TASK_ID finished."