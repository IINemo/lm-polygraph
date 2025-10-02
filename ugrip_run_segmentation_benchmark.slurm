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
#SBATCH --time=24:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

mkdir -p slurm_logs/$SLURM_JOB_ID

MODEL="ugrip_llama_instruct_vllm"
# MODEL="ugrip_gemma_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
SAMPLE_SIZE=10
TOTAL_START_TIME=$SECONDS


# experiment info
echo "Running on host: $(hostname)"
echo "Job Name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "Config file: $CONFIG_FILE_NAME"
echo "Model: $MODEL, Dataset: $DATASET, Sample Size: $SAMPLE_SIZE"

# reasoning only
echo "====STARTING REASONING ANALYSIS===="
TASK1_START_TIME=$SECONDS
scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml \
  model=$MODEL \
  dataset=$DATASET \
  subsample_eval_dataset=$SAMPLE_SIZE \
  model.load_model_args.gpu_memory_utilization=0.60  
TASK1_ELAPSED_TIME=$(($SECONDS - $TASK1_START_TIME))
echo "====FINISHED REASONING ANALYSIS===="

# answer only
echo "====STARTING ANSWER ANALYSIS===="
TASK2_START_TIME=$SECONDS
scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_segmentation_answer.yaml \
  model=$MODEL \
  dataset=$DATASET \
  subsample_eval_dataset=$SAMPLE_SIZE \
  model.load_model_args.gpu_memory_utilization=0.60  
TASK2_ELAPSED_TIME=$(($SECONDS - $TASK2_START_TIME))
echo "====FINISHED ANSWER ANALYSIS===="

echo "====BENCHMARK RUNTIME STATISTICS===="
echo "Reasoning analysis: $(($TASK1_ELAPSED_TIME / 3600)) hrs, $((($TASK1_ELAPSED_TIME / 60) % 60)) min, $(($TASK1_ELAPSED_TIME % 60)) sec"
echo "Answer analysis: $(($TASK2_ELAPSED_TIME / 3600)) hrs, $((($TASK2_ELAPSED_TIME / 60) % 60)) min, $(($TASK2_ELAPSED_TIME % 60)) sec"