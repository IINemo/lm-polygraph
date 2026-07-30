#!/bin/bash
#SBATCH --job-name=qwen35_4b_gsm8k
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/l/users/artem.shelmanov/artemvazhentsev/slurm/logs/%x-%j.out
#SBATCH --error=/l/users/artem.shelmanov/artemvazhentsev/slurm/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT=/home/artem.shelmanov/artemvazhentsev/lm-polygraph
PYTHON=/l/users/artem.shelmanov/artemvazhentsev/conda/envs/lm-polygraph-main/bin/python
HF_CACHE=/l/users/artem.shelmanov/hf_cache
OUTPUT_ROOT=/l/users/artem.shelmanov/artemvazhentsev/lm-polygraph/workdir

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_XET_CACHE="$HF_CACHE/xet"
export TORCH_HOME="$HF_CACHE/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 

cd "$PROJECT"

HYDRA_CONFIG="$PROJECT/examples/configs/polygraph_eval_gsm8k_simple_instruct.yaml" \
  "$PYTHON" -u "$PROJECT/scripts/polygraph_eval" \
    model=qwen-no-thinking \
    model.path=Qwen/Qwen3.5-4B \
    subsample_eval_dataset=10 \
    stat_calculators.1.cfg.subsample_train_dataset=50 \
    stat_calculators.1.cfg.subsample_background_train_dataset=50 \
    cache_path="$OUTPUT_ROOT"
