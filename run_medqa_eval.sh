#!/bin/bash
# Command to run polygraph_eval with MedQA dataset and llama3-1-8b-instruct model

cd /common/home/yl2310/lm-polygraph

# Run polygraph_eval with the MedQA config
python3 scripts/polygraph_eval \
    --config-path=examples/configs \
    --config-name=polygraph_eval_medqa_llama









