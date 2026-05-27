#!/bin/bash
#SBATCH --job-name=ugrip_direct      # Job name
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


# Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / Falcon3-7B-Instruct on direct
# datasets (no '### Answer:' reasoning split, slicing_target=null inherited
# from polygraph_eval_ugrip.yaml). Estimators: ugrip_benchmark_estimators_lite
# (all metrics except TokenSAR and Focus, both commented out in that file).
# HF whitebox (CausalLM) so attention saving on this branch works.


for MODEL in ugrip_llama_instruct ugrip_qwen25_instruct ugrip_falcon3_instruct; do
  for DATASET in UGRIP-LM-Polygraph/gsm8k-direct UGRIP-LM-Polygraph/mmlu-direct UGRIP-LM-Polygraph/medmcqa-direct; do
    SPLIT=test
    FN=process_output_mcq
    [[ "$DATASET" == *medmcqa* ]] && SPLIT=validation
    [[ "$DATASET" == *gsm8k* ]] && FN=process_output_number
    uv run --python 3.11 scripts/polygraph_eval \
      --config-dir=./examples/configs \
      --config-name=polygraph_eval_ugrip.yaml \
      model=$MODEL \
      dataset=$DATASET \
      generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
      estimators=ugrip_benchmark_estimators_lite.yaml \
      process_output_fn.fn_name=$FN \
      use_claim_ue=false \
      +output_attentions=false \
      subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT
  done
done


echo "Waiting for all benchmark runs to complete..."
wait

echo "All jobs finished at $(date)"
