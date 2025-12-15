# Configuration Parameters Documentation

This document describes all parameters available in `polygraph_eval` configuration files (e.g., `polygraph_eval_coqa.yaml`).

## Hydra Configuration

### `hydra.run.dir`
- **Type**: String (template)
- **Description**: Directory path where Hydra will save experiment outputs. Uses template variables like `${cache_path}`, `${task}`, `${model.path}`, `${dataset}`, and datetime formatting.
- **Example**: `${cache_path}/${task}/${model.path}/${dataset}/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- **Default**: Set by Hydra

### `defaults`
- **Type**: List
- **Description**: List of default configuration files to include. These are merged with the current config.
- **Common values**:
  - `model: bloomz-560m` - Model configuration
  - `estimators: default_estimators` - UE method estimators
  - `stat_calculators: default_calculators` - Statistics calculators
  - `base_processing_*` - Dataset-specific processing configs
  - `_self_` - Include current config file

## Path and Output Configuration

### `cache_path`
- **Type**: String
- **Description**: Base directory path for storing experiment outputs and cache files.
- **Example**: `./workdir/output`
- **Default**: `./workdir/output`

### `save_path`
- **Type**: String (template)
- **Description**: Directory where evaluation results will be saved. Can use Hydra template variables.
- **Example**: `'${hydra:run.dir}'` (uses Hydra's run directory)
- **Default**: Set by Hydra's `run.dir`

## Task Configuration

### `task`
- **Type**: String
- **Description**: Task type identifier. Affects which generation metrics are automatically included.
- **Possible values**:
  - `"qa"` - Question answering (default for CoQA, TriviaQA, etc.)
  - `"ats"` - Abstractive text summarization (uses AlignScore with `target_is_claims=False`)
  - `"nmt"` - Neural machine translation (adds COMET metric)
- **Default**: `"qa"`

### `instruct`
- **Type**: Boolean
- **Description**: Whether the model is instruction-tuned. Affects how prompts are formatted.
- **Example**: `false` for base models, `true` for instruction-tuned models
- **Default**: `false`

## Dataset Configuration

### `dataset`
- **Type**: List or String
- **Description**: Dataset identifier. Can be:
  - Hugging Face dataset: `['LM-polygraph/coqa', 'continuation']` (repo name and config)
  - Local CSV file path: `'./data/my_dataset.csv'`
- **Example**: `['LM-polygraph/coqa', 'continuation']`
- **Required**: Yes

### `text_column`
- **Type**: String
- **Description**: Name of the column in the dataset containing input/prompt texts.
- **Example**: `"input"`
- **Default**: `"input"`

### `label_column`
- **Type**: String or null
- **Description**: Name of the column containing target/reference texts. Set to `null` if no ground truth available.
- **Example**: `"output"`
- **Default**: `null` (optional)

### `train_split`
- **Type**: String
- **Description**: Name of the training split in the dataset (used for few-shot examples).
- **Example**: `"train"`
- **Default**: `"train"`

### `eval_split`
- **Type**: String
- **Description**: Name of the evaluation/test split to use for evaluation.
- **Example**: `"test"`
- **Default**: `"test"`

### `load_from_disk`
- **Type**: Boolean
- **Description**: If `true`, loads dataset from local disk using `datasets.load_from_disk()` instead of downloading from Hugging Face.
- **Example**: `false`
- **Default**: `false`

### `trust_remote_code`
- **Type**: Boolean
- **Description**: Whether to trust remote code when loading datasets from Hugging Face (required for some datasets with custom loading scripts).
- **Example**: `false`
- **Default**: `false`

### `size`
- **Type**: Integer or null
- **Description**: Maximum number of samples to use from the dataset. If `null`, uses the entire dataset.
- **Example**: `10000` or `null`
- **Default**: `null`

## Few-Shot Learning Configuration

### `n_shot`
- **Type**: Integer
- **Description**: Number of few-shot examples to include in prompts. Set to `0` for zero-shot evaluation.
- **Example**: `0` (zero-shot) or `5` (5-shot)
- **Default**: `5`

### `few_shot_split`
- **Type**: String
- **Description**: Dataset split to use for selecting few-shot examples.
- **Example**: `"train"`
- **Default**: `"train"`

### `few_shot_prompt`
- **Type**: String or null
- **Description**: Custom prompt template for few-shot examples. If `null`, uses default formatting.
- **Example**: `null` or `"Question: {input}\nAnswer: {output}\n\n"`
- **Default**: `null`

## Generation Configuration

### `max_new_tokens`
- **Type**: Integer
- **Description**: Maximum number of new tokens to generate per input.
- **Example**: `20` (short answers) or `128` (longer responses)
- **Default**: `100`

### `generation_params`
- **Type**: Dictionary
- **Description**: Additional parameters for text generation. See `GenerationParameters` class for full list.
- **Common parameters**:
  - `stop_strings`: List of strings that stop generation when encountered
  - `temperature`: Sampling temperature (0.0-2.0)
  - `top_k`: Top-k sampling
  - `top_p`: Nucleus sampling
  - `do_sample`: Whether to use sampling vs greedy
  - `num_beams`: Number of beams for beam search
  - `repetition_penalty`: Penalty for repetition (1.0 = no penalty)
- **Example**:
  ```yaml
  generation_params:
    stop_strings:
      - "\n"
    temperature: 0.7
  ```
- **Default**: `{}` (empty dict)

## Evaluation Configuration

### `subsample_eval_dataset`
- **Type**: Integer
- **Description**: Number of samples to use from the evaluation dataset. Set to `-1` to use the entire dataset.
- **Example**: `100` (use 100 samples) or `-1` (use all)
- **Default**: `-1`

### `batch_size`
- **Type**: Integer
- **Description**: Number of samples to process in each batch during evaluation.
- **Example**: `1` (one at a time) or `8` (batch of 8)
- **Default**: `1`
- **Note**: Larger batch sizes are faster but require more memory.

### `seed`
- **Type**: List of Integers
- **Description**: Random seeds to use for reproducibility. Evaluation runs once per seed.
- **Example**: `[1]` (single run) or `[1, 2, 3]` (three runs with different seeds)
- **Default**: `[1]`

## Metrics Configuration

### `generation_metrics`
- **Type**: List or null
- **Description**: Custom list of generation quality metrics. If `null`, uses default metrics based on task type.
- **Default metrics** (when `null`):
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BLEU
  - BERTScore
  - SBERT similarity
  - Accuracy
  - AlignScore (task-dependent)
  - COMET (for NMT tasks)
- **Custom format**:
  ```yaml
  generation_metrics:
    - name: RougeMetric
      args: ["rouge1"]
    - name: BLEUMetric
      kwargs: {}
  ```
- **Default**: `null` (uses task-specific defaults)

## Error Handling

### `ignore_exceptions`
- **Type**: Boolean
- **Description**: If `true`, continues evaluation even when individual samples fail. If `false`, stops on first error.
- **Example**: `false`
- **Default**: `false`

## Advanced Configuration

### `hf_cache`
- **Type**: String or null
- **Description**: Custom Hugging Face cache directory. If `null`, uses `HF_HOME` environment variable or default location.
- **Example**: `"/path/to/cache"`
- **Default**: `null` (uses environment variable)

### `hf_token`
- **Type**: String or null
- **Description**: Hugging Face API token for accessing private models/datasets. Can also be set via `HF_TOKEN` environment variable.
- **Example**: `"hf_xxxxxxxxxxxx"`
- **Default**: `null`

### `cache_path` (for HF_DATASETS_OFFLINE)
- **Type**: String
- **Description**: Cache directory for datasets when `HF_DATASETS_OFFLINE=1` environment variable is set.
- **Note**: Different from the main `cache_path` parameter above.
- **Default**: Not used unless `HF_DATASETS_OFFLINE=1`

### `deberta_batch_size`
- **Type**: Integer
- **Description**: Batch size for DeBERTa-based NLI models used in some uncertainty estimation methods.
- **Example**: `10`
- **Default**: `10`

### `language`
- **Type**: String
- **Description**: Language code for multilingual models and metrics. Affects which DeBERTa model is used.
- **Example**: `"en"`, `"ru"`, `"zh"`
- **Default**: `"en"`

### `use_claim_ue`
- **Type**: Boolean
- **Description**: Whether to use claim-level uncertainty estimation. Adds ROC-AUC and PR-AUC metrics.
- **Example**: `false`
- **Default**: `false`

### `multiref`
- **Type**: Boolean
- **Description**: Whether the dataset has multiple reference answers per question. Wraps metrics in `AggregatedMetric`.
- **Example**: `true` (for TruthfulQA with multiple answers)
- **Default**: `false`

## Processing Functions

### `process_output_fn`
- **Type**: Dictionary or null
- **Description**: Custom function to preprocess model outputs before metric calculation.
- **Format**:
  ```yaml
  process_output_fn:
    path: "path/to/script.py"
    fn_name: "function_name"
  ```
- **Default**: `null`

### `process_target_fn`
- **Type**: Dictionary or null
- **Description**: Custom function to preprocess target/reference texts before metric calculation.
- **Format**: Same as `process_output_fn`
- **Default**: `null`

## Model-Specific Parameters

These are typically defined in separate model config files (referenced in `defaults`):

- `model.path`: Hugging Face model identifier or local path
- `model.type`: Model type (`Whitebox`, `Blackbox`, `VisualLM`, `vLLMCausalLM`)
- `model.load_model_args`: Arguments for model loading (e.g., `device_map`)
- `model.path_to_load_script`: Custom model loading script (optional)

## Estimator Configuration

These are typically defined in separate estimator config files (referenced in `defaults`):

- `estimators`: List of uncertainty estimation methods to evaluate
- Each estimator can have a `name` and optional `cfg` dictionary

## Stat Calculator Configuration

These are typically defined in separate stat calculator config files (referenced in `defaults`):

- `stat_calculators`: List of statistics calculators (can include `"auto"` for defaults)
- Each calculator can have `name`, `cfg`, `stats`, `dependencies`, and `builder`

## Example: Complete Config Breakdown

```yaml
# Hydra output directory template
hydra:
  run:
    dir: ${cache_path}/${task}/${model.path}/${dataset}/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Include default configs
defaults:
  - model: bloomz-560m              # Model config
  - estimators: default_estimators # UE methods
  - stat_calculators: default_calculators # Stats
  - base_processing_coqa           # Dataset processing
  - _self_                         # This file

# Paths
cache_path: ./workdir/output        # Base output directory
save_path: '${hydra:run.dir}'       # Results save location

# Task
task: qa                            # Question answering task
instruct: false                     # Not instruction-tuned

# Dataset
dataset: ['LM-polygraph/coqa', 'continuation']  # HF dataset
text_column: input                  # Input column name
label_column: output                 # Target column name
train_split: train                  # Training split
eval_split: test                    # Evaluation split
n_shot: 0                           # Zero-shot (no few-shot examples)
few_shot_split: train               # Where to get few-shot examples
few_shot_prompt: null               # Default prompt format
trust_remote_code: false            # Don't trust remote code
size: null                          # Use full dataset

# Generation
max_new_tokens: 20                  # Generate up to 20 tokens
load_from_disk: false               # Download from HF
generation_params:
  stop_strings:                     # Stop at newline
    - "\n"

# Evaluation
subsample_eval_dataset: -1          # Use all samples
batch_size: 1                       # Process one at a time
generation_metrics: null            # Use default metrics
ignore_exceptions: false           # Stop on errors
seed:                               # Single run with seed 1
    - 1
```

## Notes

- Most parameters have sensible defaults and are optional
- Parameters can be overridden via command line: `polygraph_eval ... model.path=meta-llama/Llama-3.1-8B`
- Some parameters are only relevant for specific model types or tasks
- Check the corresponding model/estimator/stat_calculator config files for additional parameters


