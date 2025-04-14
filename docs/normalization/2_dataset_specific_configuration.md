# Dataset-Specific Normalization Configurations in LM-Polygraph

## Overview
Dataset-specific normalization configurations in LM-Polygraph allow fine-tuning how uncertainty scores are normalized for different tasks and data types. These configurations can be found in the evaluation config files under `/examples/configs/` and its subfolders.

## Configuration Structure

### 1. Common Parameters

Every dataset-specific configuration includes these core normalization parameters:

```yaml
# Dataset sampling configuration
subsample_background_train_dataset: 1000  # Size of background dataset for normalization
subsample_train_dataset: 1000             # Size of task-specific calibration dataset 
subsample_eval_dataset: -1                # Size of evaluation dataset (-1 = full)

# Training data settings
train_dataset: null                       # Optional separate training dataset
train_test_split: false                   # Whether to split data for calibration
test_split_size: 1                        # Test split ratio if splitting enabled

# Background dataset configuration 
background_train_dataset: allenai/c4      # Default background dataset
background_train_dataset_text_column: text # Text column name
background_train_dataset_label_column: url # Label column name
background_load_from_disk: false          # Loading mode
```

### 2. Task-Specific Configurations

#### Question-Answering Tasks (TriviaQA, MMLU, CoQA)
```yaml
# Additional QA-specific settings
process_output_fn:
  path: output_processing_scripts/qa_normalize.py
  fn_name: normalize_qa
normalize: true
normalize_metrics: true
target_ignore_regex: null
```

#### Translation Tasks (WMT)
```yaml
# Translation-specific normalization
source_ignore_regex: "^.*?: "            # Regex to clean source text
target_ignore_regex: null                # Regex to clean target text
normalize_translations: true
```

#### Summarization Tasks (XSum, AESLC)
```yaml
# Summarization normalization
normalize_summaries: true
output_ignore_regex: null
processing:
  trim_outputs: true
  lowercase: true
```

### 3. Language-Specific Settings

For multilingual tasks (especially in claim-level fact-checking):

```yaml
# Language-specific normalization
language: "en"  # Options: en, zh, ar, ru
multilingual_normalization:
  enabled: true
  use_language_specific_bins: true
  combine_language_statistics: false
```

## Usage Examples

### 1. Basic QA Task Configuration
```yaml
hydra:
  run:
    dir: ${cache_path}/${task}/${model.path}/${dataset}/${now:%Y-%m-%d}

defaults:
  - model: default
  - _self_

dataset: triviaqa
subsample_train_dataset: 1000
normalize: true
process_output_fn:
  path: output_processing_scripts/triviaqa.py
  fn_name: normalize_qa
```

### 2. Translation Task Setup
```yaml
dataset: wmt14_deen
subsample_train_dataset: 2000
source_ignore_regex: "^Translation: "
normalize_translations: true
background_train_dataset: null
```

### 3. Multilingual Configuration
```yaml
dataset: person_bio
language: zh
multilingual_normalization:
  enabled: true
  use_language_specific_bins: true
subsample_train_dataset: 1000
background_train_dataset: allenai/c4
```

## Key Considerations

### 1. Dataset Size and Sampling
- Use `subsample_train_dataset` to control calibration dataset size
- Larger values provide better calibration but increase compute time
- Default value of 1000 works well for most tasks

### 2. Background Dataset Usage
- Background dataset provides additional calibration data
- Useful for tasks with limited in-domain data
- C4 dataset is default choice for English tasks

### 3. Processing and Cleaning
- Task-specific normalization functions handle special cases
- Regular expressions clean input/output texts
- Language-specific processing for multilingual tasks

### 4. Performance Impact
- Larger sample sizes increase normalization quality but computational cost
- Background dataset usage adds overhead
- Consider caching normalized values for repeated evaluations

## Best Practices

1. **Dataset Size Selection**
   - Use at least 1000 samples for calibration
   - Increase for complex tasks or when accuracy is critical
   - Consider computational resources available

2. **Background Dataset Usage**
   - Use for tasks with limited training data
   - Ensure background data distribution matches task
   - Consider language and domain compatibility

3. **Processing Configuration**
   - Configure task-specific normalization functions
   - Use appropriate regex patterns for cleaning
   - Enable language-specific processing for multilingual tasks

4. **Optimization Tips**
   - Cache normalized values when possible
   - Use smaller sample sizes during development
   - Enable background dataset loading from disk for large datasets