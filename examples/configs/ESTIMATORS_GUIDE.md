# How to Control Which Uncertainty Estimation Methods Run

By default, `polygraph_eval` runs all methods defined in `default_estimators.yaml`. Here are several ways to limit which methods are evaluated:

## Method 1: Use a Different Estimator Config File

Change the `estimators` reference in your config file's `defaults` section:

```yaml
defaults:
  - model: bloomz-560m
  - estimators: tutorial_estimators  # Use a smaller set of methods
  - stat_calculators: default_calculators
  - base_processing_coqa
  - _self_
```

Available estimator config files:
- `default_estimators` - All methods (100+ estimators)
- `tutorial_estimators` - Smaller subset for tutorials
- `default_estimators_vllm` - Methods compatible with vLLM
- `default_estimators_visual` - Methods for visual models
- `default_claim_estimators` - Claim-level methods only

## Method 2: Override Estimators Directly in Config

You can override the estimators list directly in your config file:

```yaml
defaults:
  - model: bloomz-560m
  - estimators: default_estimators  # Still reference it, but override below
  - stat_calculators: default_calculators
  - base_processing_coqa
  - _self_

# Override estimators - only run these methods
estimators:
  - name: MaximumSequenceProbability
  - name: Perplexity
  - name: MeanTokenEntropy
  - name: SemanticEntropy
```

## Method 3: Override via Command Line

You can override the estimators config file via command line:

```bash
polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    estimators=tutorial_estimators \
    model.path=allenai/Llama-3.1-Tulu-3-8B
```

## Method 4: Create a Custom Estimator Config File

Create your own estimator config file in `examples/configs/estimators/`:

**File: `examples/configs/estimators/my_custom_estimators.yaml`**
```yaml
- name: MaximumSequenceProbability
- name: Perplexity
- name: MeanTokenEntropy
- name: SemanticEntropy
- name: SAR
```

Then use it in your config:
```yaml
defaults:
  - estimators: my_custom_estimators
```

Or via command line:
```bash
polygraph_eval ... estimators=my_custom_estimators
```

## Method 5: Override Individual Estimators via Command Line

You can also override the entire estimators list via command line (though this is more complex):

```bash
polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    estimators=[{name:MaximumSequenceProbability},{name:Perplexity}]
```

## Estimator Configuration Format

Each estimator in the list can have:
- `name`: The estimator class name (required)
- `cfg`: Configuration dictionary (optional)

Example with configuration:
```yaml
estimators:
  - name: LexicalSimilarity
    cfg:
      metric: "rouge1"
  - name: EigValLaplacian
    cfg:
      similarity_score: "NLI_score"
      affinity: "entail"
  - name: Focus
    cfg:
      model_name: '${model.path}'
      path: "${cache_path}/focus/${model.path}/token_idf.pkl"
      gamma: 0.9
      p: 0.01
```

## Common Estimator Names

Here are some commonly used estimators you can include:

**Information-based (fast, low compute):**
- `MaximumSequenceProbability`
- `Perplexity`
- `MeanTokenEntropy`
- `SelfCertainty`
- `MeanPointwiseMutualInformation`

**Semantic/M diversity (medium compute):**
- `SemanticEntropy`
- `SAR`
- `TokenSAR`
- `SentenceSAR`
- `NumSemSets`
- `EigValLaplacian`
- `LUQ`

**Density-based (requires training data, low compute):**
- `MahalanobisDistanceSeq`
- `RelativeMahalanobisDistanceSeq`
- `RDESeq`
- `PPLMDSeq`

**Sampling-based (high compute):**
- `MonteCarloSequenceEntropy`
- `MonteCarloNormalizedSequenceEntropy`
- `PTrueSampling`

**Reflexive:**
- `PTrue`

## Example: Quick Test with Only 3 Methods

Create a minimal config for quick testing:

**File: `examples/configs/estimators/quick_test.yaml`**
```yaml
- name: MaximumSequenceProbability
- name: Perplexity
- name: MeanTokenEntropy
```

Then run:
```bash
polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    estimators=quick_test \
    model.path=allenai/Llama-3.1-Tulu-3-8B \
    subsample_eval_dataset=100
```

## Notes

- Some estimators require specific stat calculators (e.g., NLI-based methods need DeBERTa)
- Some estimators only work with whitebox models (not blackbox)
- Some estimators require training data (density-based methods)
- Check the estimator documentation for requirements

