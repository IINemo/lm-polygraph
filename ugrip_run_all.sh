# Logging output  
LOG_DIR="hyperpod/$(date +"%Y-%m-%d")"
LOG_FILE="${LOG_DIR}/benchmark_output_$(date +"%m-%d-%y_%H-%M").log"


mkdir -p "${LOG_DIR}"


{
  START_TIME=$(date +%s)
  echo "Starting benchmarking jobs at $(date)"

  # CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval --config-dir ./examples/configs --config-name polygraph_eval_ugrip model=ugrip_gemma_instruct_vllm.yaml dataset=UGRIP-LM-Polygraph/gsm8k-reasoning subsample_eval_dataset=2 &

  CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval --config-dir ./examples/configs --config-name polygraph_eval_ugrip \
    hydra.mode=MULTIRUN model=ugrip_gemma_instruct_vllm.yaml \
    dataset=UGRIP-LM-Polygraph/mmlu-reasoning \
    subsample_eval_dataset=10 &


  echo "Waiting for benchmarking jobs to finish..."
  wait


  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  HOURS=$((DURATION / 3600))
  MINUTES=$(( (DURATION % 3600) / 60 ))
  SECONDS=$((DURATION % 60))


  echo "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"


  echo "All benchmarking jobs finished at $(date)"
} 2>&1 | tee "${LOG_FILE}"
