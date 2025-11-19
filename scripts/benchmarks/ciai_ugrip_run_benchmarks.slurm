#!/bin/bash
#SBATCH --job-name=test             # Job name
#SBATCH --output=output.%A_%a.txt   # Standard output and error log      # slurm_logs/%x-%j.out
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=45G                   # Total RAM to be used
#SBATCH --cpus-per-task=10          # Number of CPU cores
#SBATCH --gres=gpu:2                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=24:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

# python temp.py

#  UGRIP-LM-Polygraph/gsm8k-direct
#  UGRIP-LM-Polygraph/gsm8k-reasoning
#  UGRIP-LM-Polygraph/medmcqa-direct
#  UGRIP-LM-Polygraph/medmcqa-reasoning
#  UGRIP-LM-Polygraph/mmlu-direct
#  UGRIP-LM-Polygraph/mmlu-reasoning


# python3 --version

nvidia-smi
hostname


mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/gsm8k-reasoning subsample_eval_dataset=3
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/mmlu-reasoning generation_metrics=ugrip_generation_metrics_temp.yaml estimators=ugrip_benchmark_estimators_lite.yaml subsample_eval_dataset=1000 batch_size=50
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/medmcqa-reasoning subsample_eval_dataset=1000 eval_split=validation


# Only accuracy, MSP, PPL, MTE
scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/gsm8k-reasoning generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml estimators=ugrip_benchmark_estimators_lite.yaml subsample_eval_dataset=1000 batch_size=20
scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/mmlu-reasoning generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml estimators=ugrip_benchmark_estimators_lite.yaml subsample_eval_dataset=1000 batch_size=50
scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml  model=ugrip_gemma_instruct_vllm dataset=UGRIP-LM-Polygraph/medmcqa-reasoning generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml estimators=ugrip_benchmark_estimators_lite.yaml subsample_eval_dataset=1000 batch_size=50 eval_split=validation


echo "Waiting for all benchmark runs to complete..."
wait 

echo "All jobs finished at $(date)"