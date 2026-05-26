#!/bin/bash
set -e

nvidia-smi
hostname


mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
source .venv/bin/activate


# Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / Falcon3-7B-Instruct on direct
# datasets (no '### Answer:' reasoning split, slicing_target=null inherited
# from polygraph_eval_ugrip.yaml). Estimators: ugrip_benchmark_estimators_lite
# (all metrics except TokenSAR and Focus, both commented out in that file).
# HF whitebox (CausalLM) so attention saving on this branch works.


# llama
for DATASET in UGRIP-LM-Polygraph/gsm8k-direct UGRIP-LM-Polygraph/mmlu-direct UGRIP-LM-Polygraph/medmcqa-direct; do
  SPLIT=test
  [[ "$DATASET" == *medmcqa* ]] && SPLIT=validation
  uv run --python 3.11 scripts/polygraph_eval \
    --config-dir=./examples/configs \
    --config-name=polygraph_eval_ugrip.yaml \
    model=ugrip_llama_instruct \
    dataset=$DATASET \
    generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
    estimators=ugrip_benchmark_estimators_lite.yaml \
    subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT
done


# qwen2.5
for DATASET in UGRIP-LM-Polygraph/gsm8k-direct UGRIP-LM-Polygraph/mmlu-direct UGRIP-LM-Polygraph/medmcqa-direct; do
  SPLIT=test
  [[ "$DATASET" == *medmcqa* ]] && SPLIT=validation
  uv run --python 3.11 scripts/polygraph_eval \
    --config-dir=./examples/configs \
    --config-name=polygraph_eval_ugrip.yaml \
    model=ugrip_qwen25_instruct \
    dataset=$DATASET \
    generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
    estimators=ugrip_benchmark_estimators_lite.yaml \
    subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT
done


# falcon3
for DATASET in UGRIP-LM-Polygraph/gsm8k-direct UGRIP-LM-Polygraph/mmlu-direct UGRIP-LM-Polygraph/medmcqa-direct; do
  SPLIT=test
  [[ "$DATASET" == *medmcqa* ]] && SPLIT=validation
  uv run --python 3.11 scripts/polygraph_eval \
    --config-dir=./examples/configs \
    --config-name=polygraph_eval_ugrip.yaml \
    model=ugrip_falcon3_instruct \
    dataset=$DATASET \
    generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
    estimators=ugrip_benchmark_estimators_lite.yaml \
    subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT
done


echo "All jobs finished at $(date)"
