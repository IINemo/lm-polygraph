# LM-Polygraph Estimator Selection Guide

## Quick Decision Tree

```
Start Here
    │
    ├─ Do you have access to model internals (logits)?
    │   │
    │   ├─ YES (WhiteboxModel) ──> Do you have training data?
    │   │                           │
    │   │                           ├─ YES ──> Density-based methods
    │   │                           │          (MahalanobisDistance, RDE)
    │   │                           │
    │   │                           └─ NO ──> What aspect concerns you most?
    │   │                                      │
    │   │                                      ├─ Token probabilities ──> Perplexity, MaximumSequenceProbability
    │   │                                      ├─ Output diversity ──> TokenEntropy, SemanticEntropy
    │   │                                      └─ Self-awareness ──> PTrue
    │   │
    │   └─ NO (BlackboxModel) ──> Can you generate multiple samples?
    │                              │
    │                              ├─ YES ──> LexicalSimilarity, NumSemSets, EigValLaplacian
    │                              └─ NO ──> Limited options (Verbalized methods if supported)
    │
    └─ What level of granularity do you need?
        │
        ├─ Sequence-level (overall uncertainty) ──> Most methods
        ├─ Token-level (word-by-word) ──> TokenEntropy, MaximumTokenProbability, TokenSAR
        └─ Claim-level (fact-checking) ──> ClaimConditionedProbability, PTrueClaim
```

## Detailed Recommendations by Use Case

### 1. General Purpose Uncertainty Estimation

**For White-box Models:**
- **First choice**: `Perplexity` or `MeanTokenEntropy`
  - Fast, well-understood, no training required
  - Good correlation with human judgments
  - Works well across different domains

**For Black-box Models:**
- **First choice**: `LexicalSimilarity('rougeL')`
  - Works with any model
  - Requires multiple samples (set n=5-10)
  - Higher diversity = higher uncertainty

### 2. Hallucination Detection

**For Factual Questions:**
- **White-box**: `PTrue` - Asks model if its answer is true
- **Black-box**: `SemanticEntropy` - High entropy indicates conflicting information

**For Creative/Open-ended Tasks:**
- **Any model**: `LexicalSimilarity` or `NumSemSets`
  - High variation in outputs suggests uncertainty

### 3. Real-time Applications (Need Speed)

**Fastest Methods:**
1. `Perplexity` - Single forward pass
2. `MaximumSequenceProbability` - No additional computation
3. `TokenEntropy` - Minimal overhead

**Avoid:**
- Sampling-based methods (SemanticEntropy, LexicalSimilarity with many samples)
- Ensemble methods

### 4. High-Stakes Applications (Need Accuracy)

**Best Performers (with enough data):**
1. `MahalanobisDistance` - If you have training data
2. `SemanticEntropy` - Captures meaning-level uncertainty
3. Ensemble methods (`EPTmi`, `EPSrmi`) - If you have multiple models

**Recommended Combination:**
```python
# Use multiple complementary methods
estimators = [
    Perplexity(),           # Lexical uncertainty
    SemanticEntropy(),      # Semantic uncertainty  
    PTrue()                 # Model's self-assessment
]
```

### 5. Specific Scenarios

#### Chatbots/Conversational AI
- **Primary**: `TokenEntropy` - Identify uncertain parts of responses
- **Secondary**: `LexicalSimilarity` - Detect when model is "making things up"

#### Question Answering
- **Primary**: `PTrue` or `ClaimConditionedProbability`
- **Secondary**: `MaximumSequenceProbability`

#### Code Generation
- **Primary**: `TokenEntropy` - High entropy often indicates syntax uncertainty
- **Secondary**: `Perplexity` - Unusual code patterns

#### Medical/Legal Text
- **Primary**: `MahalanobisDistance` (train on verified data)
- **Secondary**: `SemanticEntropy` with high sample count

## Method Characteristics Table

| Method | Model Type | Speed | Accuracy | Requires Training | Requires Sampling |
|--------|-----------|-------|----------|-------------------|-------------------|
| Perplexity | White-box | Fast | Good | No | No |
| TokenEntropy | White-box | Fast | Good | No | No |
| MaximumSequenceProbability | White-box | Fast | Moderate | No | No |
| PTrue | White-box | Medium | Good | No | No |
| SemanticEntropy | White-box | Slow | Excellent | No | Yes |
| LexicalSimilarity | Any | Medium | Good | No | Yes |
| NumSemSets | Any | Slow | Good | No | Yes |
| MahalanobisDistance | White-box | Fast | Excellent | Yes | No |
| EigValLaplacian | Any | Slow | Good | No | Yes |

## Practical Examples

### Example 1: Simple Uncertainty Check
```python
from lm_polygraph import WhiteboxModel
from lm_polygraph.estimators import Perplexity
from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty

model = WhiteboxModel.from_pretrained("gpt2")
estimator = Perplexity()

# Quick uncertainty check
result = estimate_uncertainty(model, estimator, "The capital of France is")
if result.uncertainty > 5.0:  # Threshold depends on model
    print("High uncertainty detected!")
```

### Example 2: Black-box Hallucination Detection
```python
from lm_polygraph import BlackboxModel
from lm_polygraph.estimators import LexicalSimilarity

model = BlackboxModel.from_openai("YOUR_KEY", "gpt-3.5-turbo")
estimator = LexicalSimilarity("rougeL")

# Generate multiple samples
result = estimate_uncertainty(
    model, estimator, 
    "What happened in the fictional Battle of Hogwarts in 1823?",
    generation_params={"temperature": 0.8, "n": 5}
)

# High uncertainty (close to 0) suggests model is making things up
if result.uncertainty > -0.3:
    print("Model seems uncertain - possible hallucination")
```

### Example 3: Token-level Analysis
```python
from lm_polygraph.estimators import TokenEntropy

estimator = TokenEntropy()
result = estimate_uncertainty(model, estimator, "Explain quantum entanglement")

# Identify uncertain tokens
tokens = result.generation_tokens
uncertainties = result.uncertainty
for token, uncertainty in zip(tokens, uncertainties):
    if uncertainty > 2.0:  # High entropy threshold
        print(f"Uncertain token: {token} (entropy: {uncertainty:.2f})")
```

## Combining Methods

For best results, combine complementary methods:

```python
def comprehensive_uncertainty(model, text):
    results = {}
    
    # Lexical uncertainty
    results['perplexity'] = estimate_uncertainty(
        model, Perplexity(), text
    ).uncertainty
    
    # Semantic uncertainty (if sampling available)
    results['semantic'] = estimate_uncertainty(
        model, SemanticEntropy(), text,
        generation_params={"do_sample": True, "num_return_sequences": 10}
    ).uncertainty
    
    # Model self-assessment
    results['p_true'] = estimate_uncertainty(
        model, PTrue(), text
    ).uncertainty
    
    # Combine scores (example: simple average)
    combined = np.mean(list(results.values()))
    
    return combined, results
```

## Rules of Thumb

1. **Start simple**: Try Perplexity or TokenEntropy first
2. **Use multiple samples**: For diversity-based methods, use 5-10 samples minimum
3. **Domain-specific tuning**: Thresholds vary by model and domain
4. **Ensemble when possible**: Multiple methods often catch different failure modes
5. **Validate empirically**: Test on your specific use case with known examples

## Common Pitfalls

1. **Using white-box methods on black-box models**: Check compatibility first
2. **Insufficient samples**: Diversity methods need multiple generations
3. **Ignoring computational cost**: Some methods are too slow for real-time use
4. **Over-relying on one method**: Different methods catch different issues
5. **Not calibrating thresholds**: Uncertainty scales vary between models

Remember: No single method is perfect for all scenarios. Choose based on your specific requirements for speed, accuracy, and available resources.