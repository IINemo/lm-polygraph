# Core Normalization Configuration

## Overview
Core normalization configuration in LM-Polygraph defines how uncertainty scores are transformed into interpretable confidence values. These configurations control the fundamental behavior of all normalization methods across the system.

## Base Configuration Location
Core normalization configurations are located in:
```
/examples/configs/normalization/fit/default.yaml
```

## Available Normalization Methods

### 1. MinMax Normalization
Linearly scales uncertainty scores to [0,1] range.

```yaml
normalization:
  type: "minmax"
  clip: true  # Whether to clip values outside [0,1] range
```

### 2. Quantile Normalization 
Transforms scores into percentile ranks using empirical CDF.

```yaml
normalization:
  type: "quantile"
```

### 3. Binned Performance-Calibrated Confidence (Binned PCC)
Maps uncertainty scores to confidence bins based on output quality.

```yaml
normalization:
  type: "binned_pcc"
  params:
    num_bins: 10  # Number of bins for mapping
```

### 4. Isotonic Performance-Calibrated Confidence (Isotonic PCC)
Uses monotonic regression to map uncertainty to confidence while preserving ordering.

```yaml
normalization:
  type: "isotonic_pcc"
  params:
    y_min: 0.0  # Minimum confidence value
    y_max: 1.0  # Maximum confidence value
    increasing: false  # Whether mapping should be increasing
    out_of_bounds: "clip"  # How to handle out-of-range values
```

## Common Parameters

### Calibration Strategy
```yaml
normalization:
  calibration:
    strategy: "dataset_specific"  # or "global"
    background_dataset: null  # Optional background dataset for global calibration
```

### Data Processing
```yaml
normalization:
  processing:
    ignore_nans: true  # Whether to ignore NaN values in calibration
    normalize_metrics: true  # Whether to normalize quality metrics
```

### Caching
```yaml
normalization:
  cache:
    enabled: true
    path: "${cache_path}/normalization"
    version: "v1"
```

## Usage Examples

### Basic MinMax Normalization
```yaml
normalization:
  type: "minmax"
  clip: true
  calibration:
    strategy: "dataset_specific"
```

### Global Isotonic PCC
```yaml
normalization:
  type: "isotonic_pcc"
  params:
    y_min: 0.0
    y_max: 1.0
    increasing: false
  calibration:
    strategy: "global"
    background_dataset: "allenai/c4"
```

### Binned PCC with Custom Settings
```yaml
normalization:
  type: "binned_pcc"
  params:
    num_bins: 20
  processing:
    ignore_nans: false
    normalize_metrics: true
  cache:
    enabled: true
```

## Best Practices

1. **Method Selection**
   - Use MinMax/Quantile for simple scaling needs
   - Use PCC methods when interpretability is crucial
   - Prefer Isotonic PCC when preserving score ordering is important

2. **Calibration Strategy**
   - Use dataset-specific calibration when possible
   - Use global calibration when consistency across tasks is needed
   - Consider using background dataset for robust global calibration

3. **Performance Considerations**
   - Enable caching for large datasets
   - Adjust bin count based on dataset size
   - Monitor memory usage with large calibration sets

## Integration with Other Configs
Core normalization settings can be overridden by:
- Task-specific configs
- Model-specific configs
- Instruction-tuned model configs

Core settings serve as defaults when not specified in other configuration layers.