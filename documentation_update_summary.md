# LM-Polygraph Documentation Update Summary

## Completed Documentation Updates

### 1. Estimator Classes
Updated comprehensive docstrings for the following key estimators:

#### Information-based Methods
- **Perplexity** (`src/lm_polygraph/estimators/perplexity.py`)
  - Added detailed class and method docstrings
  - Included theory, use cases, and examples
  - Added references to relevant papers

- **TokenEntropy & MeanTokenEntropy** (`src/lm_polygraph/estimators/token_entropy.py`)
  - Documented both token-level and sequence-level variants
  - Explained Shannon entropy calculation
  - Added usage examples and cross-references

- **MaximumSequenceProbability & MaximumTokenProbability** (`src/lm_polygraph/estimators/max_probability.py`)
  - Clarified joint probability vs token probability
  - Added examples for both sequence and token levels
  - Included notes about length normalization

#### Meaning Diversity Methods
- **LabelProb** (`src/lm_polygraph/estimators/label_prob.py`)
  - Explained semantic clustering approach
  - Documented the multi-sample requirement
  - Added practical examples

- **LexicalSimilarity** (`src/lm_polygraph/estimators/lexical_similarity.py`)
  - Comprehensive documentation for black-box compatibility
  - Explained ROUGE metrics and their differences
  - Added parameter validation and error handling

#### Reflexive Methods
- **PTrue** (`src/lm_polygraph/estimators/p_true.py`)
  - Documented the self-evaluation approach
  - Added references to Kadavath et al. paper
  - Included limitations and model requirements

### 2. Model Classes
Enhanced documentation for both model types:

#### BlackboxModel (`src/lm_polygraph/utils/model.py`)
- Comprehensive class docstring explaining API support
- Detailed examples for OpenAI and HuggingFace usage
- Documented logprobs support for newer OpenAI models
- Enhanced `generate_texts` method documentation

#### WhiteboxModel (`src/lm_polygraph/utils/model.py`)
- Detailed class documentation with supported model types
- Enhanced `generate` method docs with parameter details
- Improved `generate_texts` high-level method documentation
- Comprehensive `from_pretrained` factory method docs

### 3. Model Adapters (NEW)
Added comprehensive documentation for specialized model adapters:

#### WhiteboxModelBasic (`src/lm_polygraph/model_adapters/whitebox_model_basic.py`)
- Lightweight adapter for simple use cases
- Direct pass-through to model methods
- Clear differentiation from full WhiteboxModel

#### WhiteboxModelvLLM (`src/lm_polygraph/model_adapters/whitebox_model_vllm.py`)
- High-performance vLLM integration
- Detailed performance benefits documentation
- Output format conversion explained

#### VisualWhiteboxModel (`src/lm_polygraph/model_adapters/visual_whitebox_model.py`)
- Comprehensive multimodal support documentation
- Image handling (URLs and local paths)
- VLM-specific examples and use cases

### 4. Core Functions
- **estimate_uncertainty** (`src/lm_polygraph/utils/estimate_uncertainty.py`)
  - Complete rewrite with detailed parameter descriptions
  - Added comprehensive examples for different use cases
  - Included cross-references to related functions
  - Documented return value structure

### 5. Module-Level Documentation
- Added comprehensive docstrings to `__init__.py` files:
  - `src/lm_polygraph/__init__.py` - Main package overview
  - `src/lm_polygraph/estimators/__init__.py` - Estimator categories
  - `src/lm_polygraph/model_adapters/__init__.py` - Adapter comparison

### 6. Additional Documentation Created
- **estimator_selection_guide.md** - Comprehensive guide for choosing estimators
  - Decision tree for quick selection
  - Use case recommendations
  - Method comparison table
  - Practical examples

## Documentation Standards Established

### 1. Consistent Docstring Format
- One-line summary
- Detailed description with context
- Structured parameter documentation with types
- Clear return value descriptions
- Relevant examples
- Cross-references via "See Also"
- Academic references where applicable

### 2. Example Quality
- Practical, runnable examples
- Coverage of common use cases
- Both simple and advanced usage patterns
- Output explanations

### 3. Type Annotations
- Added type hints to method signatures
- Clarified complex types (Dict, List, Union)
- Documented expected shapes for arrays

## Key Improvements Made

1. **User-Friendly**: Documentation now explains not just "what" but "why" and "when" to use each method
2. **Comprehensive Examples**: Each class has practical examples that users can adapt
3. **Cross-References**: Methods now reference related approaches
4. **Theory Integration**: Added paper references and theoretical context
5. **Error Guidance**: Better error messages and parameter validation
6. **Model Adapter Coverage**: All specialized model adapters now fully documented

## Remaining Documentation Tasks

### High Priority
1. **Statistical Calculators**: Document what each calculator computes
2. **Other Estimators**: Apply same documentation standards to remaining ~25 estimators
3. **Generation Parameters**: Document all available parameters
4. **Error Classes**: Add helpful error messages with solutions

### Medium Priority
1. **Utils Functions**: Document helper functions and utilities
2. **Manager Classes**: Document UEManager and related classes
3. **Processors**: Document data processors and transformers
4. **Metrics**: Document evaluation metrics

### Low Priority
1. **Internal Methods**: Document private/internal methods
2. **Test Files**: Add docstrings to test cases
3. **Scripts**: Document utility scripts

## Files Updated Summary

### Core Model Files
- `src/lm_polygraph/utils/model.py` - BlackboxModel, WhiteboxModel
- `src/lm_polygraph/utils/estimate_uncertainty.py` - Main estimation function

### Model Adapters
- `src/lm_polygraph/model_adapters/whitebox_model_basic.py`
- `src/lm_polygraph/model_adapters/whitebox_model_vllm.py`
- `src/lm_polygraph/model_adapters/visual_whitebox_model.py`
- `src/lm_polygraph/model_adapters/__init__.py`

### Estimators
- `src/lm_polygraph/estimators/perplexity.py`
- `src/lm_polygraph/estimators/label_prob.py`
- `src/lm_polygraph/estimators/token_entropy.py`
- `src/lm_polygraph/estimators/max_probability.py`
- `src/lm_polygraph/estimators/p_true.py`
- `src/lm_polygraph/estimators/lexical_similarity.py`
- `src/lm_polygraph/estimators/__init__.py`

### Package Files
- `src/lm_polygraph/__init__.py`

## Impact

This documentation update significantly improves the developer experience by:
- Making all model types (standard and adapters) equally well-documented
- Providing clear guidance on when to use each component
- Offering practical examples for all major use cases
- Establishing consistent documentation standards for future contributions

The project now has comprehensive documentation for its core APIs, making it more accessible to new users while maintaining depth for advanced users.