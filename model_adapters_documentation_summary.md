# Model Adapters Documentation Update Summary

## Overview

I've added comprehensive documentation to all three model adapter files that were missing proper docstrings. These adapters provide specialized functionality beyond the standard WhiteboxModel and BlackboxModel classes.

## Files Updated

### 1. `src/lm_polygraph/model_adapters/whitebox_model_basic.py`

**Purpose**: Simplified adapter for basic uncertainty estimation workflows

**Documentation Added**:
- Comprehensive module docstring explaining the lightweight nature
- Detailed class docstring highlighting differences from main WhiteboxModel
- Method documentation for all public methods:
  - `__init__`: Parameter descriptions with examples
  - `generate`: Direct pass-through functionality explained
  - `tokenize`: Token processing with default args
  - `generate_texts`: High-level generation (noted non-standard interface)
  - `device`: Device retrieval
  - `__call__`: Forward pass for logits

**Key Features Documented**:
- Minimal overhead design
- Direct model method pass-through
- No custom stopping criteria
- Lighter weight for simple use cases

### 2. `src/lm_polygraph/model_adapters/whitebox_model_vllm.py`

**Purpose**: High-performance adapter using vLLM inference engine

**Documentation Added**:
- Module docstring explaining vLLM benefits
- Extensive class docstring covering:
  - vLLM performance advantages
  - Continuous batching and PagedAttention
  - Log probability access for uncertainty
  - Usage examples with vLLM
- Detailed method documentation:
  - `__init__`: vLLM setup and parameter handling
  - `generate`: HuggingFace-to-vLLM format conversion
  - `post_processing`: Output format transformation
  - `tokenize`: Input preparation
  - `generate_texts`: High-level text generation

**Key Features Documented**:
- High-throughput inference capabilities
- Automatic output format conversion
- Token probability preservation
- GPU optimization benefits

### 3. `src/lm_polygraph/model_adapters/visual_whitebox_model.py`

**Purpose**: Vision-language model support for multimodal uncertainty estimation

**Documentation Added**:
- Module docstring for multimodal context
- Comprehensive class docstring including:
  - Supported VLM types (BLIP-2, LLaVA, etc.)
  - Image handling capabilities
  - Multimodal uncertainty estimation
  - Multiple usage examples
- Method documentation:
  - `__init__`: Image loading and setup
  - `generate`: Multimodal generation with scores
  - `generate_texts`: High-level image+text generation
  - `from_pretrained`: Factory method with examples

**Key Features Documented**:
- Combined image-text processing
- Support for URL and local images
- Compatibility with VLM architectures
- Batch processing considerations

### 4. `src/lm_polygraph/model_adapters/__init__.py`

**Documentation Added**:
- Module overview explaining adapter purposes
- Quick comparison of available adapters
- Usage examples for each adapter type
- Proper `__all__` exports

## Documentation Standards Applied

1. **Comprehensive Examples**: Each adapter includes practical usage examples
2. **Parameter Details**: All parameters documented with types and descriptions
3. **Cross-References**: Links to related classes via "See Also" sections
4. **Architecture Notes**: Explained design decisions and use cases
5. **Compatibility Info**: Clear requirements (e.g., vLLM installation)

## Key Improvements

1. **Clarity**: Users can now understand when to use each adapter
2. **Discoverability**: The module docstring helps users find the right adapter
3. **Integration Guidance**: Examples show how to use with uncertainty estimators
4. **Performance Notes**: Documentation includes performance characteristics

## Usage Patterns Documented

### WhiteboxModelBasic
```python
# For simple, lightweight usage
model = WhiteboxModelBasic(base_model, tokenizer, tokenizer_args)
```

### WhiteboxModelvLLM
```python
# For high-performance inference
llm = LLM(model="meta-llama/Llama-2-7b-hf")
model = WhiteboxModelvLLM(llm, sampling_params)
```

### VisualWhiteboxModel
```python
# For vision-language tasks
model = VisualWhiteboxModel.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    "VisualLM",
    image_paths=["image.jpg"]
)
```

## Impact

These documentation updates:
- Make the adapters more accessible to users
- Clarify the specific use cases for each adapter
- Provide clear migration paths from standard models
- Enable better decision-making about which adapter to use
- Support the overall LM-Polygraph documentation improvement initiative

The model adapters are now fully documented with the same high standards as the main model classes, making the entire model interface consistent and well-documented.