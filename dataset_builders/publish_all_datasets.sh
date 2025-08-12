#!/usr/bin/env bash
set -euo pipefail

# Publish all supported datasets with the new stripped_input column
# Namespace is fixed to rvashurin per request.

NAMESPACE="rvashurin"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set. Export it and retry." >&2
  exit 1
fi

DATASETS=(
  # base
  trivia_qa_tiny
  aeslc

  # babi_qa
  babi_qa
    
  # coqa
  coqa
  coqa_empirical_baselines
  coqa_ling_1s
  coqa_verb_1s_top1
  coqa_verb_1s_topk
  coqa_verb_2s_cot
  coqa_verb_2s_top1
  coqa_verb_2s_topk
  coqa_simple_instruct
    
  # gsm8k
  gsm8k
  gsm8k_simple_instruct

  # mmlu
  mmlu
  mmlu_empirical_baselines
  mmlu_ling_1s
  mmlu_verb_1s_top1
  mmlu_verb_1s_topk
  mmlu_verb_2s_cot
  mmlu_verb_2s_top1
  mmlu_verb_2s_topk
  mmlu_simple_instruct

  # person
  person_bio_ar
  person_bio_en
  person_bio_ru
  person_bio_zh

  # triviaqa
  triviaqa
  triviaqa_empirical_baselines
  triviaqa_ling_1s
  triviaqa_verb_1s_top1
  triviaqa_verb_1s_topk
  triviaqa_verb_2s_cot
  triviaqa_verb_2s_top1
  triviaqa_verb_2s_topk
  triviaqa_simple_instruct

  # wiki
  wiki_bio

  # wmt
  wmt14_deen
  wmt14_fren
  wmt14_fren_simple_instruct
  wmt19_deen
  wmt19_deen_simple_instruct
  wmt19_ruen
  wmt19_ruen_simple_instruct

  # truthfulqa
  truthfulqa
  truthfulqa_simple_instruct

  # samsum
  samsum
  samsum_simple_instruct

  # xsum
  xsum
  xsum_simple_instruct
)

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Resolve Python from current environment (allows overriding with PYTHON=...)
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python &>/dev/null; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Could not find a python interpreter in PATH." >&2
    exit 1
  fi
fi

for ds in "${DATASETS[@]}"; do
  echo "=== Publishing $ds to namespace ${NAMESPACE} ==="
  "$PYTHON_BIN" "${ROOT_DIR}/manager.py" \
    --dataset "$ds" \
    --publish \
    --namespace "${NAMESPACE}"
done

echo "All datasets processed."
