#!/bin/bash
#SBATCH --job-name=falcon_direct     # Job name
#SBATCH --output=output.%A_%a.txt    # Standard output and error log
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --mem=200G                   # Total RAM to be used
#SBATCH --cpus-per-task=10           # Number of CPU cores
#SBATCH --gres=gpu:4                 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                # Use the gpu partition
#SBATCH --time=24:00:00              # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos           # To enable the use of up to 8 GPUs


nvidia-smi
hostname


mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
source .venv/bin/activate


# Falcon-7b-instruct (HF whitebox, attention saving enabled) on direct datasets.
# Uses polygraph_eval_ugrip_falcon_direct.yaml which:
#   - selects model=ugrip_falcon_instruct (HF, not vLLM)
#   - sets generate_until=['\nUser', '\n\nUser:'] so Falcon's chat-template
#     turn marker does not leak into the output
#   - inherits slicing_target=null (direct datasets have no '### Answer:' split)


# gsm8k
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/gsm8k-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_estimators_lite.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=test

uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/gsm8k-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_tokensar_estimator.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=test


# mmlu
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/mmlu-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_estimators_lite.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=test

uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/mmlu-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_tokensar_estimator.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=test


# medmcqa
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/medmcqa-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_estimators_lite.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=validation

uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip_falcon_direct.yaml \
  dataset=UGRIP-LM-Polygraph/medmcqa-direct \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=ugrip_benchmark_tokensar_estimator.yaml \
  subsample_eval_dataset=1000 batch_size=1 eval_split=validation


echo "Waiting for all benchmark runs to complete..."
wait

echo "All jobs finished at $(date)"
