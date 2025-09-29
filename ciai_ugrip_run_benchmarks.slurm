#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=output.%A_%a.txt   # Standard output and error log      # slurm_logs/%x-%j.out
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=10          # Number of CPU cores
#SBATCH --gres=gpu:2                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

# hostname

# python temp.py

mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/gsm8k-reasoning subsample_eval_dataset=5

echo "Waiting for all benchmark runs to complete..."
wait 

echo "All jobs finished at $(date)"