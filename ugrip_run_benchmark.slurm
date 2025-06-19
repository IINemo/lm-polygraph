#!/bin/bash

#SBATCH --job-name=ugrip_polygraph_benchmarks
#SBATCH --output=slurm_logs/%x-%j.out 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Create directory for slurm logs if it doesn't 
# exist yet
mkdir -p slurm_logs

echo "Setting up conda environment..."
module load miniconda3
conda activate reasoning_uq

echo "Beginning UGRIP Benchmark: 2 models, 2 GPUs"
echo "Job started on $(hostname) at $(date)"

# First benchmark: Llama 3.1 8b Instruct
echo "Starting Llama-3.1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python ./scripts/polygraph_eval \
    --config-name polygraph_eval_ugrip \
    --multirun \
    model=ugrip_llama_instruct \
    dataset=ugrip_gsm8k_direct,ugrip_gsm8k_reasoning,ugrip_mmlu_direct,ugrip_mmlu_reasoning,ugrip_medmcqa_direct,ugrip_medmcqa_reasoning &

# Second benchmark: Gemma 3 12b Instruct
echo "Starting Gemma-2 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python ./scripts/polygraph_eval \
    --config-name polygraph_eval_ugrip \
    --multirun \
    model=ugrip_gemma_instruct \
    dataset=ugrip_gsm8k_direct,ugrip_gsm8k_reasoning,ugrip_mmlu_direct,ugrip_mmlu_reasoning,ugrip_medmcqa_direct,ugrip_medmcqa_reasoning &

echo "Waiting for all benchmark runs to complete..."
wait 

echo "All jobs finished at $(date)"