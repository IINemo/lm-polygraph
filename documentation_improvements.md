# LM-Polygraph Documentation Improvement Suggestions

## Executive Summary

After reviewing the LM-Polygraph project documentation, I've identified several areas where documentation can be enhanced to improve developer experience, maintainability, and user adoption. The project has a solid foundation with good API documentation via Sphinx, comprehensive examples, and a detailed README, but there are opportunities for improvement.

## 1. Missing Docstrings in Source Code

### Issue
Many classes and methods in the source code lack proper docstrings, making it harder for developers to understand the code and for automated documentation tools to generate comprehensive API docs.

### Examples of files needing docstrings:
- `src/lm_polygraph/estimators/perplexity.py` - The `Perplexity` class lacks class and method docstrings
- `src/lm_polygraph/estimators/label_prob.py` - No documentation at all
- Many other estimator classes likely have similar issues

### Recommendation
Add comprehensive docstrings following the NumPy or Google docstring style for all:
- Classes (with description, attributes, and examples)
- Methods (with parameters, returns, raises, and examples)
- Module-level documentation

### Specific Code Examples

#### Example 1: Improving `perplexity.py`
```python
"""Perplexity-based uncertainty estimation for language models."""

import numpy as np
from typing import Dict
from .estimator import Estimator


class Perplexity(Estimator):
    """
    Perplexity-based uncertainty estimator for language models.
    
    Calculates the perplexity of generated sequences as a measure of uncertainty.
    Lower perplexity indicates higher confidence in the generation. Perplexity is
    the exponential of the average negative log-likelihood of the tokens.
    
    This estimator is particularly useful for:
    - Quick uncertainty estimation with low computational overhead
    - Comparing relative uncertainties across different inputs
    - Detecting potentially problematic generations
    
    Attributes:
        dependencies: List of required statistics, includes 'greedy_log_likelihoods'
        level: Estimation level, set to 'sequence'
    
    References:
        Fomicheva et al., 2020. "Unsupervised Quality Estimation for Neural Machine
        Translation" (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/)
    
    Examples:
        >>> from lm_polygraph.estimators import Perplexity
        >>> from lm_polygraph.utils.model import WhiteboxModel
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = Perplexity()
        >>> uncertainty = estimate_uncertainty(model, estimator, "What is AI?")
        >>> print(f"Perplexity score: {uncertainty.uncertainty}")
    """
    
    def __init__(self):
        """Initialize the Perplexity estimator with required dependencies."""
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self) -> str:
        """Return the string identifier for this estimator."""
        return "Perplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate perplexity scores for the given statistics.
        
        Perplexity is calculated as the negative mean of log likelihoods,
        which represents the average uncertainty per token.
        
        Parameters:
            stats: Dictionary containing 'greedy_log_likelihoods' key with
                   log likelihood arrays for each sequence. Each array contains
                   log probabilities for tokens in the generated sequence.
                   
        Returns:
            Array of perplexity scores (negative mean log likelihood)
            for each sequence. Higher values indicate higher uncertainty.
            
        Raises:
            KeyError: If 'greedy_log_likelihoods' is not in stats
            ValueError: If log likelihoods are empty or malformed
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.mean(ll) for ll in log_likelihoods])
```

#### Example 2: Improving `label_prob.py`
```python
"""Label probability uncertainty estimation based on semantic clustering."""

import numpy as np
from typing import Dict
from .estimator import Estimator


class LabelProb(Estimator):
    """
    Label probability uncertainty estimator based on semantic clustering.
    
    This estimator measures uncertainty by analyzing the distribution of
    generated samples across semantic clusters. Higher concentration in
    a single cluster indicates lower uncertainty.
    
    The method works by:
    1. Generating multiple samples for the same input
    2. Clustering samples by semantic similarity (done by stat calculator)
    3. Computing uncertainty as 1 - (largest_cluster_size / total_samples)
    
    This approach captures semantic uncertainty beyond surface-level variations,
    making it effective for detecting when a model is uncertain about the
    core meaning of its response.
    
    Attributes:
        dependencies: Requires 'semantic_classes_entail' statistics
        level: Operates at 'sequence' level
        
    Examples:
        >>> from lm_polygraph.estimators import LabelProb
        >>> estimator = LabelProb()
        >>> # Requires sampling-based generation
        >>> uncertainty = estimate_uncertainty(model, estimator, "Explain quantum physics")
    """
    
    def __init__(self):
        """Initialize LabelProb with semantic clustering dependency."""
        super().__init__(["semantic_classes_entail"], "sequence")

    def __str__(self) -> str:
        """Return the string identifier for this estimator."""
        return "LabelProb"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty based on semantic cluster distribution.
        
        Parameters:
            stats: Dictionary containing 'semantic_classes_entail' with:
                   - 'class_to_sample': mapping from class indices to sample indices
                   - 'sample_to_class': mapping from sample indices to class indices
                   
        Returns:
            Array of uncertainty scores in [0, 1] range where:
            - 0 indicates all samples belong to one semantic class (low uncertainty)
            - Values close to 1 indicate uniform distribution across classes (high uncertainty)
            
        Raises:
            KeyError: If required semantic clustering data is not in stats
        """
        batch_class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        batch_sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        ues = []
        for batch_i, class_to_sample in batch_class_to_sample.items():
            num_samples = len(batch_sample_to_class[batch_i])
            largest_class_size = max([len(samples) for samples in class_to_sample])
            # Uncertainty is the complement of the largest class proportion
            ues.append(1 - largest_class_size / num_samples)

        return np.array(ues)
```

## 2. Enhanced API Documentation Structure

### Current State
The Sphinx documentation auto-generates from docstrings but lacks:
- Conceptual overviews for each module
- Cross-references between related methods
- Visual diagrams explaining the architecture

### Recommendations
1. Add module-level documentation files explaining:
   - The purpose and theory behind each estimator category
   - When to use which estimator
   - Performance characteristics

2. Create architecture diagrams showing:
   - How estimators, calculators, and models interact
   - The data flow in the uncertainty estimation pipeline
   - The relationship between different uncertainty methods

3. Add a "Quick Start" section to docs with common use cases

## 3. README Improvements

### Current State
The README is comprehensive but could be enhanced for better navigation and clarity.

### Recommendations
1. Add a table of contents at the top for easier navigation
2. Create a "Which Uncertainty Method Should I Use?" decision tree or guide
3. Add performance benchmarks comparing methods
4. Include troubleshooting section for common issues
5. Add links to academic papers for each method in the overview table

## 4. Example Notebooks Enhancement

### Current State
Good examples exist but could be more educational.

### Recommendations
1. Add markdown cells explaining:
   - The theory behind each uncertainty method demonstrated
   - When and why to use specific methods
   - Interpretation of results

2. Create specialized examples:
   - "Detecting Hallucinations in Medical Text"
   - "Uncertainty Estimation for Code Generation"
   - "Comparing Uncertainty Methods on Your Dataset"

3. Add performance profiling examples showing memory and compute usage

## 5. Developer Documentation

### Missing Components
1. **Architecture Guide**: Document the overall system design, key abstractions, and extension points
2. **Plugin Development Guide**: How to create custom estimators, calculators, and metrics
3. **Testing Guide**: How to write tests for new components
4. **Performance Optimization Guide**: Best practices for efficient uncertainty estimation

### Recommendation Structure
Create a `docs/developer/` directory with:
- `architecture.md`
- `creating_estimators.md`
- `testing_guide.md`
- `performance_guide.md`

## 6. Type Hints and Documentation

### Issue
Some methods lack complete type hints, making IDE support and documentation less effective.

### Recommendation
Add comprehensive type hints throughout the codebase, especially for:
- Public API methods
- Return types of complex functions
- Callback and configuration parameters

## 7. Configuration Documentation

### Missing
Clear documentation on all available configuration options for:
- Generation parameters
- Estimator-specific parameters
- Benchmark configurations

### Recommendation
Create a `docs/configuration.md` file documenting all parameters with:
- Default values
- Valid ranges
- Impact on performance
- Examples

## 8. API Reference Improvements

### Current State
Auto-generated from docstrings but lacks organization.

### Recommendations
1. Group methods by functionality rather than alphabetically
2. Add a "Common Workflows" section showing method combinations
3. Include inheritance diagrams for estimator classes
4. Add "See Also" sections linking related methods

## 9. Changelog and Migration Guides

### Missing
- Detailed changelog with breaking changes highlighted
- Migration guides between versions

### Recommendation
1. Create `CHANGELOG.md` following Keep a Changelog format
2. Add migration guides for major version changes
3. Document deprecation warnings properly

## 10. Web Demo Documentation

### Issue
The web demo is marked as obsolete but documentation still references it.

### Recommendation
1. Update all references to clarify the demo is unsupported
2. Document alternatives or plans for a new demo
3. Remove or clearly mark obsolete demo documentation

## 11. Inline Code Documentation

### Recommendations
1. Add comments for complex algorithms explaining the mathematical operations
2. Document magic numbers and thresholds with references
3. Add TODO/FIXME comments with issue tracker links

## 12. Documentation Build and Deployment

### Recommendations
1. Add documentation build status badge to README
2. Document how to build docs locally in CONTRIBUTING.md
3. Set up documentation preview for pull requests
4. Add documentation linting to CI/CD pipeline

## 13. Utils Module Documentation

### Issue
The utils module contains critical functionality like the `estimate_uncertainty` function and model classes, but documentation could be more comprehensive.

### Specific Improvements Needed

#### 1. Improve `estimate_uncertainty` function documentation
The current docstring is good but could include:
- More detailed parameter descriptions
- Common pitfalls and error cases
- Performance considerations
- Links to related estimators

#### 2. Model Classes Documentation
Both `WhiteboxModel` and `BlackboxModel` need:
- More detailed class-level documentation
- Better examples showing different initialization methods
- Clear documentation of method limitations
- Migration guide from raw HuggingFace models

#### Example improvement for `WhiteboxModel`:
```python
class WhiteboxModel(Model):
    """
    White-box model wrapper for HuggingFace models with full access to logits and hidden states.
    
    This class provides a unified interface for uncertainty estimation methods that require
    access to model internals such as token probabilities, attention weights, and hidden states.
    It supports both encoder-decoder and decoder-only architectures.
    
    Key features:
    - Full access to model logits and probabilities
    - Support for custom generation parameters
    - Efficient batch processing
    - Compatible with all white-box uncertainty estimation methods
    
    Attributes:
        model: The underlying HuggingFace model
        tokenizer: HuggingFace tokenizer for the model
        model_path: Path or identifier of the model
        model_type: Type of model architecture ('CausalLM' or 'Seq2Seq')
        generation_parameters: Default parameters for text generation
        
    Examples:
        Basic initialization:
        >>> from lm_polygraph import WhiteboxModel
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        
        With custom parameters:
        >>> model = WhiteboxModel.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     device_map="auto",
        ...     load_in_8bit=True,
        ...     generation_params={"temperature": 0.7, "max_new_tokens": 100}
        ... )
        
        Using existing HuggingFace objects:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = WhiteboxModel(base_model, tokenizer)
        
    See Also:
        BlackboxModel: For models without logit access
        estimate_uncertainty: Main function for uncertainty estimation
    """
```

## 14. Statistical Calculators Documentation

### Issue
The stat_calculators module is crucial for the framework but lacks comprehensive documentation explaining:
- What each calculator computes
- Dependencies between calculators
- Performance implications
- When to use custom calculators

### Recommendation
Create detailed documentation for each stat calculator with:
- Mathematical formulation
- Computational complexity
- Memory requirements
- Example usage

## 15. Error Messages and Troubleshooting

### Current State
Error messages could be more informative for common issues.

### Recommendations
1. Create custom exception classes with helpful messages:
```python
class IncompatibleEstimatorError(Exception):
    """Raised when an estimator is incompatible with the model type."""
    def __init__(self, estimator_name, model_type, required_type):
        super().__init__(
            f"Estimator '{estimator_name}' requires a {required_type} model, "
            f"but got {model_type}. "
            f"For black-box models, consider using estimators like: "
            f"LexicalSimilarity, NumSemSets, or EigValLaplacian."
        )
```

2. Add a troubleshooting guide in docs covering:
   - Common installation issues
   - GPU memory errors
   - Model compatibility problems
   - Performance optimization tips

## Implementation Priority

### High Priority
1. Add missing docstrings to all public APIs
2. Create "Which Method to Use" guide
3. Update obsolete web demo references
4. Add type hints to public methods

### Medium Priority
1. Create developer documentation
2. Add conceptual overviews to docs
3. Enhance example notebooks with explanations
4. Create configuration documentation

### Low Priority
1. Add architecture diagrams
2. Create specialized example notebooks
3. Add performance profiling examples
4. Set up documentation preview system

## Actionable Next Steps

### Documentation Sprint Checklist

#### Week 1: Critical Documentation
- [ ] Add docstrings to all estimator classes in `src/lm_polygraph/estimators/`
- [ ] Document all public methods in `WhiteboxModel` and `BlackboxModel`
- [ ] Update README to clarify web demo status
- [ ] Create a quick "Which Estimator Should I Use?" guide

#### Week 2: Developer Experience
- [ ] Add type hints to all public APIs
- [ ] Create `CHANGELOG.md` with version history
- [ ] Write developer guide for creating custom estimators
- [ ] Document all configuration parameters

#### Week 3: User Guides
- [ ] Enhance example notebooks with theory explanations
- [ ] Create troubleshooting guide
- [ ] Add performance benchmarking guide
- [ ] Document best practices for production use

#### Week 4: Infrastructure
- [ ] Set up documentation linting in CI/CD
- [ ] Configure automatic API documentation generation
- [ ] Create documentation style guide
- [ ] Add contribution guidelines for documentation

### Documentation Standards to Adopt

1. **Docstring Format**: Use NumPy style consistently
2. **Code Examples**: Include at least one example per public class/function
3. **Type Hints**: Required for all public APIs
4. **Cross-References**: Link related methods and classes
5. **Version Info**: Document when features were added/deprecated

### Metrics for Success

- **Coverage**: 100% of public APIs have docstrings
- **Examples**: Every estimator has a usage example
- **Clarity**: New users can choose appropriate estimators without external help
- **Completeness**: All error messages are informative and actionable

## Quick Wins (Can be done immediately)

1. **Add module-level docstrings** to all `__init__.py` files explaining the purpose of each submodule
2. **Create a FAQ section** in the README addressing common questions
3. **Add "Since version X.Y" annotations** to new features
4. **Include computational complexity** (O notation) for each estimator
5. **Add a glossary** defining key terms (uncertainty, entropy, perplexity, etc.)

## Long-term Documentation Vision

1. **Interactive Documentation**: Jupyter notebooks that can be run in Binder/Colab
2. **Video Tutorials**: Short videos explaining key concepts
3. **API Playground**: Web interface to test estimators without installation
4. **Community Examples**: Section for user-contributed use cases
5. **Automated Benchmarks**: Daily runs showing estimator performance on standard datasets

By implementing these documentation improvements, LM-Polygraph will become more accessible to new users, easier to contribute to, and more reliable for production deployments. The focus should be on making the "happy path" obvious while providing depth for advanced users.