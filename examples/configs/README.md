# Root-level configuration reference for `polygraph_eval`

This document describes the available top-level configuration parameters for the `polygraph_eval` script, as illustrated by the example YAMLs in this directory (excluding any `person_bio*` files). For each parameter, we reference where it is consumed in the codebase.

---

## Hydra and defaults

Hydra is used to manage runtime configuration. The example configs all include a `hydra` block and a `defaults` list to select model, estimators, and statistical calculators.

```yaml
hydra:
  run:
    dir: ${cache_path}/${task}/${model.path}/${dataset}/${now:%Y-%m-%d}/${now:%H-%M-%S}
defaults:
  - model: bloomz-560m
  - estimators: default_estimators
  - stat_calculators: default_calculators
  - [...]
```
【F:examples/configs/polygraph_eval_xsum.yaml†L1-L10】

## Model parameters

The `model` Hydra group provides configuration for selecting and loading your language model. Base fields are defined in `examples/configs/model/default.yaml`【F:examples/configs/model/default.yaml†L1-L5】, and can be overridden by specific presets (e.g., `bloomz-560m.yaml`【F:examples/configs/model/bloomz-560m.yaml†L4-L10】, `vllm.yaml`【F:examples/configs/model/vllm.yaml†L4-L12】). Key parameters include:

| Name                       | Type            | Purpose                                                                                                              |
|----------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------|
| `model.path`               | `str`           | Hugging Face model identifier or path to a local model.                                                              |
| `model.type`               | `str`           | Model wrapper type (`CausalLM`, `Seq2SeqLM`, `vLLMCausalLM`).                                                         |
| `model.path_to_load_script`| `str`           | Path to the Python script implementing `load_model` and `load_tokenizer` (relative to this directory).               |
| `model.load_model_args`    | `map`           | Keyword arguments passed to `load_model` (e.g., `device_map` for HF or `gpu_memory_utilization` for vLLM).            |
| `model.load_tokenizer_args`| `map`           | Keyword arguments passed to `load_tokenizer` (e.g., `add_bos_token`).                                                |
| `model.logprobs`           | `int` (optional)| Number of logprobs to request (vLLM only).                                                                           |
| `model.device`             | `str` (optional)| Device override for vLLM (e.g., `cuda`).                                                                              |

_Note: ensemble-related parameters (`ensemble`, `mc`, `mc_seeds`, `dropout_rate`) are deprecated and omitted here._

Example presets:
```yaml
# examples/configs/model/bloomz-560m.yaml
path: bigscience/bloomz-560m
type: CausalLM
path_to_load_script: model/default_causal.py

load_model_args:
  device_map: auto
load_tokenizer_args: {}
```
【F:examples/configs/model/bloomz-560m.yaml†L4-L10】

```yaml
# examples/configs/model/vllm.yaml
path: facebook/opt-350m
type: vLLMCausalLM
path_to_load_script: model/default_vllm.py
logprobs: 20

load_model_args:
  gpu_memory_utilization: 0.8

device: cuda
```
【F:examples/configs/model/vllm.yaml†L4-L12】

## Common parameters

These parameters are defined in all of the `polygraph_eval_*.yaml` examples:

| Name                    | Type             | Purpose                                                                                                                                                                  |
|-------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cache_path`            | `str`            | Base path for caching intermediate artifacts, used as `cache_dir` when `HF_DATASETS_OFFLINE=1` and passed to certain estimators (e.g., OpenAIFactCheck).               |
| `save_path`             | `str`            | Directory where outputs (e.g., UEManager state, metrics) are saved, defaults to Hydra run dir.                                                                            |
| `instruct`              | `bool`           | Passed to dataset loader to enable instruction-style prompts (currently accepted but unused by the loader).                                                             |
| `task`                  | `str`            | Defines the task type (`nmt`, `qa`, `ats`) and affects which generation quality metrics get added (e.g., COMET for `nmt`, AlignScore for `ats`).                                            |
| `dataset`               | `str` or `[ ]`   | HuggingFace dataset identifier or path (plus subset) for evaluation.                                                                                                     |
| `text_column`           | `str`            | Name of the input column in the dataset.                                                                                                                                |
| `label_column`          | `str`            | Name of the target/output column in the dataset (if any).                                                                                                                |
| `train_split`           | `str`            | Name of the split to draw training examples from (for few-shot and training-stat calculators).                                                                            |
| `eval_split`            | `str`            | Name of the split to draw evaluation examples from.                                                                                                                     |
| `n_shot`                | `int`            | Number of few-shot examples to prepend (passed to `Dataset.load`).                                                                                                       |
| `few_shot_split`        | `str`            | Which split to sample few-shot examples from.                                                                                                                            |
| `few_shot_prompt`       | `str` or `null`  | Template for formatting few-shot examples into the prompt.                                                                                                              |
| `load_from_disk`        | `bool`           | If true, loads a locally cached HF dataset from disk instead of downloading.                                                                                            |
| `trust_remote_code`     | `bool`           | Passed to `datasets.load_dataset` to allow execution of remote code.                                                                                                      |
| `max_new_tokens`        | `int`            | Number of tokens to generate per example.                                                                                                                                |
| `generation_params`     | `map`            | Overrides for generation decoding parameters (see [GenerationParameters] for details).                                                                                   |
| `subsample_eval_dataset`| `int`            | If ≥0, subsamples the eval dataset to this size (or fraction if <1); set to `-1` to disable subsampling.                                                                 |
| `generation_metrics`    | `list` or `null` | Custom metric definitions (overrides the defaults).                                                                                                                      |
| `ignore_exceptions`     | `bool`           | If true, continues the evaluation run upon individual-sample errors.                                                                                                      |
| `batch_size`            | `int`            | Batch size for dataset iteration.                                                                                                                                       |
| `seed`                  | `list[int]`      | Random seed(s) to use for dataset sampling and model seeding (multiple seeds allow repeated runs).                                                                        |
| `multiref`              | `bool`           | Wraps each generation metric in `AggregatedMetric` for multi-reference evaluation (only for QA tasks).                                                                   |
| `source_ignore_regex`   | `str`            | Pattern to strip from source inputs (used for COMET metric in `nmt`).                                                                                                    |
| `target_ignore_regex`   | `str`            | Regex to filter target text before accuracy/metrics.                                                                                                         |
| `output_ignore_regex`   | `str`            | Regex to filter generated text before accuracy/metrics.                                                                                                      |
| `normalize`             | `bool`           | If true, applies simple whitespace and punctuation normalization before metrics (mutually exclusive with `process_*_fn`).                                                 |

### Claim-level UE parameters

| Name                    | Type             | Purpose                                                                                                                                                                  |
|-------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_claim_ue`          | `bool`           | If true, adds claim-based UE metrics (ROCAUC, PRAUC) and OpenAI fact-checking metric.                                                                                   |
| `language`              | `str`            | Language code passed to fact-checking/calculators (default: `"en"`).                                                                                                  |
| `n_threads`             | `int`            | Number of threads for OpenAIFactCheck.                                                                                                                                    |

【F:examples/configs/polygraph_eval_wmt14_fren.yaml†L11-L42】【F:scripts/polygraph_eval†L68-L83】【F:scripts/polygraph_eval†L90-L115】

## Generation parameters

The `generation_params` block allows fine-grained control over decoding. It maps directly to the `GenerationParameters` dataclass:

```python
@dataclass
class GenerationParameters:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    generate_until: tuple = ()
    allow_newlines: bool = True

# See src/lm_polygraph/utils/generation_parameters.py for details
```
【F:src/lm_polygraph/utils/generation_parameters.py†L5-L35】

## Metric customization

By default, a suite of NLG and UE metrics is registered in `polygraph_eval`. You can override the list via `generation_metrics` or inject preprocessing hooks via the `process_output_fn` / `process_target_fn` patterns.

```yaml
generation_metrics: null
process_output_fn:
  path: instruct/output_processing_scripts/coqa.py
  fn_name: normalize_em_coqa
process_target_fn:
  path: instruct/output_processing_scripts/coqa.py
  fn_name: normalize_em_coqa
```
【F:examples/configs/base_processing_coqa.yaml†L1-L6】【F:scripts/polygraph_eval†L220-L282】

When `generation_metrics` is `null`, `polygraph_eval` builds a default list including Rouge, BLEU, BertScore, SBERT, Accuracy, AlignScore (with task-specific settings), plus COMET for `nmt` and optional OpenAI fact-checking if `use_claim_ue=true`:
```python
# default generation metrics builder
RougeMetric(...), BLEUMetric(), BertScoreMetric(...), SbertMetric(), AccuracyMetric(...), AlignScore(...)
if task=="nmt": Comet(...)
if use_claim_ue: OpenAIFactCheck(...)
```
【F:scripts/polygraph_eval†L217-L248】【F:scripts/polygraph_eval†L236-L244】

The optional `multiref=true` wraps metrics in `AggregatedMetric` for multiple-reference evaluation.
【F:scripts/polygraph_eval†L284-L287】

## Dataset sampling

`polygraph_eval` can subsample the evaluation set for faster iteration:

```python
if args.subsample_eval_dataset != -1:
    dataset.subsample(args.subsample_eval_dataset, seed)
```
【F:scripts/polygraph_eval†L114-L115】

## Exception handling

By setting `ignore_exceptions=true`, any per-sample exceptions during generation or UE calculation will be caught and logged, allowing the run to continue.
【F:scripts/polygraph_eval†L135-L146】

---

*This reference was auto-derived from the example configs in this directory and the `polygraph_eval` code in* `scripts/polygraph_eval` *and* `src/lm_polygraph`.
