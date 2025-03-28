# LM-Polygraph Normalization: Impact Areas and Default Behaviors

## Normalization Impact Areas

### 1. Score Transformation
- **Raw Uncertainty Scores**
  - Original uncertainty estimates in unbounded ranges
  - Higher values indicate more uncertainty
  - Various scales depending on estimation method
  
- **Normalized Confidence Values** 
  - Bounded in [0,1] range
  - Higher values indicate more confidence
  - Directly interpretable probabilities
  - Preserves relative ordering of original scores (for Isotonic PCC)

### 2. Evaluation Pipeline
- **Calibration Stage**
  - Uses calibration dataset to learn normalization parameters
  - Requires generation quality metrics for Performance-Calibrated Confidence
  - Can use either task-specific or background data
  - Parameters are saved for reuse

- **Inference Stage** 
  - Applies learned normalization to new uncertainty estimates
  - No additional model inference required
  - Fast transformation using stored parameters
  - Can be applied to any compatible uncertainty estimator

### 3. Quality Metrics Integration
- **Metric Normalization**
  - Quality metrics are normalized to [0,1] range
  - Enables consistent calibration across different metrics
  - Handles various metric types (ROUGE, BLEU, accuracy, etc.)
  - Supports both bounded and unbounded metrics

- **Metric Selection**
  - Different metrics for different tasks
  - Task-specific normalization of quality scores
  - Multiple metrics can be used simultaneously
  - Quality metrics guide confidence calibration

### 4. Model Types Support
- **White-box Models**
  - Access to internal probabilities and logits
  - Can normalize token-level uncertainties
  - Supports both sequence and token-level calibration
  - Works with HuggingFace models

- **Black-box Models**
  - Limited to output-based uncertainty estimation
  - Only sequence-level normalization 
  - Compatible with API-based models (OpenAI, etc.)
  - No access to internal model states

## Default Behaviors

### 1. Score Processing

```python
# Default score processing behavior
normalize_scores = {
    'clip_values': True,           # Clip to [0,1] range
    'flip_uncertainty': True,      # Convert uncertainty to confidence
    'preserve_order': True,        # Maintain sample ordering
    'handle_nans': 'ignore'        # Skip NaN values in calibration
}
```

### 2. Calibration Settings

```python
# Default calibration configuration
calibration_defaults = {
    'strategy': 'dataset_specific',    # Use task-specific calibration
    'num_samples': 1000,               # Default calibration set size
    'background_data': None,           # No background data by default
    'split_ratio': None,               # No train/test split
    'cache_enabled': True              # Cache calibration parameters
}
```

### 3. Method Selection

```python
# Default normalization method selection
method_defaults = {
    'primary_method': 'isotonic_pcc',  # Default to Isotonic PCC
    'fallback_method': 'minmax',       # Use MinMax as fallback
    'combine_methods': False,          # Don't combine multiple methods
    'quality_metric': 'auto'           # Auto-select appropriate metric
}
```

### 4. Task-Specific Defaults

```yaml
# Task-type specific defaults
task_defaults:
  qa:
    metric: 'accuracy'
    normalize_answers: true
    ignore_case: true
    
  translation:
    metric: 'bleu'
    normalize_translations: true
    source_cleaning: true
    
  summarization:
    metric: 'rouge'
    normalize_summaries: true
    trim_outputs: true
```

### 5. Error Handling

```python
# Default error handling behavior
error_handling = {
    'invalid_scores': 'skip',          # Skip invalid uncertainty scores
    'missing_metrics': 'error',        # Raise error for missing metrics
    'calibration_fails': 'fallback',   # Use fallback method if calibration fails
    'out_of_bounds': 'clip'           # Clip out-of-bounds values
}
```

### 6. Memory Management

```python
# Default memory management settings
memory_settings = {
    'cache_location': '~/.cache/lm-polygraph/norm',
    'max_cache_size': '1GB',
    'clear_cache_on_exit': False,
    'compression': True
}
```

## Usage Guidelines

1. **Choosing Calibration Data**
   - Use task-specific data when available
   - Ensure calibration set is representative
   - Consider using background data for sparse tasks
   - Monitor calibration set size vs. performance

2. **Method Selection**
   - Start with Isotonic PCC for best balance
   - Use MinMax for simple scaling needs
   - Consider Binned PCC for interpretability
   - Evaluate multiple methods if uncertain

3. **Error Handling**
   - Monitor normalization failures
   - Validate calibration success
   - Check normalized score distributions
   - Verify quality metric calculations

4. **Performance Optimization**
   - Enable caching for repeated use
   - Adjust calibration set size as needed
   - Use appropriate quality metrics
   - Monitor memory usage