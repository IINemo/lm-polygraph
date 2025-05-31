# Instruction-Tuned Model Normalization Configurations in LM-Polygraph

## Overview
Instruction-tuned model configurations in LM-Polygraph provide specialized normalization settings for models that have been fine-tuned on instruction data. These configurations are located in `/examples/configs/instruct/` and include specific processing scripts and parameters for handling instruction-formatted inputs and outputs.

## Configuration Structure

### 1. Base Processing Configuration
Located in `/examples/configs/instruct/`, base processing configs define foundational normalization settings:

```yaml
# Base processing for instruction-tuned models
process_output_fn:
  path: instruct/output_processing_scripts/default.py
  fn_name: normalize_em
process_target_fn:
  path: instruct/output_processing_scripts/default.py
  fn_name: normalize_em
```

### 2. Task-Specific Processing

#### CoQA Processing
```yaml
# CoQA-specific instruction normalization
process_output_fn:
  path: instruct/output_processing_scripts/coqa.py
  fn_name: normalize_em_coqa
process_target_fn:
  path: instruct/output_processing_scripts/coqa.py
  fn_name: normalize_em_coqa
```

#### TriviaQA Processing
```yaml
# TriviaQA-specific instruction normalization
process_output_fn:
  path: instruct/output_processing_scripts/triviaqa.py
  fn_name: normalize_em_triviaqa
process_target_fn:
  path: instruct/output_processing_scripts/triviaqa.py
  fn_name: normalize_em_triviaqa
```

### 3. Processing Types

#### Chain-of-Thought (CoT) Processing
```yaml
# CoT processing settings
cot_processing:
  enabled: true
  extract_final_answer: true
  normalize_reasoning: false
  ignore_intermediate_steps: true
```

#### Top-K Processing
```yaml
# Top-K response processing
topk_processing:
  enabled: true
  k: 4  # Number of alternatives to consider
  aggregate_method: "max"  # How to combine multiple predictions
```

#### Top-1 Processing
```yaml
# Top-1 response processing
top1_processing:
  enabled: true
  normalize_confidence: true
  extract_probability: true
```

## Model-Specific Configurations

### 1. Model Type Settings
```yaml
defaults:
  - model: default_causal.py
  - _self_

model:
  type: "CausalLM"
  path_to_load_script: model/default_causal.py
  generation_params:
    do_sample: false
    num_beams: 1
    temperature: 1.0
```

### 2. Specialized Model Examples

#### Mistral Configuration
```yaml
model:
  path: mistral-7b-instruct-v0.2
  type: "CausalLM"
  load_model_args:
    device_map: auto
    trust_remote_code: true
  load_tokenizer_args:
    trust_remote_code: true
```

#### StableLM Configuration
```yaml
model:
  path: stablelm-2-12b-chat
  type: "CausalLM"
  load_model_args:
    device_map: auto
    use_flash_attention: true
```

## Integration Features

### 1. Processing Pipeline Integration
- Custom normalization functions for instruction-formatted outputs
- Task-specific answer extraction
- Confidence score normalization

### 2. Model Output Processing
- Handling of structured instruction outputs
- Extraction of final answers from reasoning chains
- Normalization of multiple response formats

### 3. Configuration Inheritance
- Base processing settings inheritance
- Task-specific overrides
- Model-specific adaptations

## Best Practices

### 1. Processing Function Selection
- Use task-specific normalizers when available
- Fall back to default processors for general cases
- Consider instruction format when selecting processors

### 2. Confidence Handling
- Enable confidence normalization for compatible models
- Configure appropriate aggregation methods for multiple outputs
- Consider model-specific confidence scales

### 3. Chain-of-Thought Processing
- Enable for models trained with CoT
- Configure appropriate answer extraction
- Consider preservation of reasoning steps

### 4. Performance Optimization
- Enable caching for processed outputs
- Configure batch processing when possible
- Balance processing complexity with performance needs

## Example Configurations

### 1. Basic Instruction Model Setup
```yaml
defaults:
  - model: default_causal
  - _self_

process_output_fn:
  path: instruct/output_processing_scripts/default.py
  fn_name: normalize_em

top1_processing:
  enabled: true
  normalize_confidence: true
```

### 2. CoT Model Configuration
```yaml
defaults:
  - model: mistral-instruct
  - _self_

cot_processing:
  enabled: true
  extract_final_answer: true

process_output_fn:
  path: instruct/output_processing_scripts/cot.py
  fn_name: normalize_cot
```

### 3. Multi-Task Model Setup
```yaml
defaults:
  - model: stablelm-chat
  - _self_

process_output_fn:
  path: instruct/output_processing_scripts/multi_task.py
  fn_name: normalize_mt

topk_processing:
  enabled: true
  k: 4
  aggregate_method: "max"
```

## Common Issues and Solutions

### 1. Output Format Mismatches
- Problem: Model outputs don't match expected instruction format
- Solution: Configure custom processing functions
- Example: Use task-specific normalizers

### 2. Confidence Scale Differences
- Problem: Different models use different confidence scales
- Solution: Enable confidence normalization
- Example: Configure model-specific scaling

### 3. Processing Pipeline Conflicts
- Problem: Multiple processing steps interfering
- Solution: Configure processing order
- Example: Set priority for different normalizers

### 4. Performance Bottlenecks
- Problem: Slow processing of instruction outputs
- Solution: Enable caching and batch processing
- Example: Configure appropriate batch sizes