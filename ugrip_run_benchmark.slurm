#!/bin/bash

#SBATCH --job-name=ugrip_polygraph_benchmarks
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --partition=gpu-1
#SBATCH --account=ugrip-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Create directory for slurm logs if it doesn't 
# exist yet
mkdir -p slurm_logs

echo "Setting up conda environment..."
module load miniconda3
conda activate reasoning_uq

cd ~/workspace/lm-polygraph

scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml model=ugrip_gemma_instruct

# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml \
# #                        process_output_fn.fn_name=your_function \
#                         model=ugrip_gemma_instruct dataset=UGRIP-LM-Polygraph/gsm8k-direct \
#                         estimators=ugrip_benchmark_estimators_lite \
#                         generation_metrics=ugrip_benchmark_generation_metrics_lite

# !!! Example:
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml model=ugrip_gemma_instruct dataset=UGRIP-LM-Polygraph/gsm8k-direct

# !!! Example: Overwriting process_output_fn or process_target_fn
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml \
#                         process_output_fn.fn_name=your_function \
#                         model=ugrip_gemma_instruct dataset=UGRIP-LM-Polygraph/gsm8k-direct

# !!! Example: Use only a subset of estimators
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml estimators=ugrip_benchmark_estimators_lite model=ugrip_llama_instruct dataset=UGRIP-LM-Polygraph/gsm8k-direct

# !!! Example: Use only a subset of generation metrics
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml generation_metrics=ugrip_benchmark_generation_metrics_lite model=ugrip_llama_instruct dataset=UGRIP-LM-Polygraph/gsm8k-direct

# !!! Example: Multirun
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml hydra.mode=MULTIRUN estimators=ugrip_benchmark_estimators_lite model=ugrip_llama_instruct,ugrip_gemma_instruct subsample_eval_dataset=10 dataset=UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning


echo "Waiting for all benchmark runs to complete..."
wait 

echo "All jobs finished at $(date)"