#!/bin/bash
#SBATCH --job-name=ugrip_answer
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH --qos=cscc-gpu-qos


echo "Running ANSWER ONLY job on host: $(hostname)"
echo "Job Name: $SLURM_JOB_NAME, Job ID: $SLURM_JOB_ID"


scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_segmentation_answer.yaml \
  model=ugrip_gemma_instruct_vllm \
  dataset=UGRIP-LM-Polygraph/gsm8k-reasoning \
  subsample_eval_dataset=10 \

echo "answer job finished."