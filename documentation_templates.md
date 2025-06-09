# LM-Polygraph Documentation Templates

This file provides standardized templates for documenting different components of the LM-Polygraph project.

## 1. Estimator Class Template

```python
"""Module-level description of what this estimator does."""

import numpy as np
from typing import Dict, List, Optional, Union
from .estimator import Estimator


class YourEstimator(Estimator):
    """
    One-line summary of the estimator.
    
    Detailed description of the uncertainty estimation method, including:
    - The theoretical basis
    - When to use this method
    - Advantages and limitations
    - Computational complexity
    
    The method works by:
    1. First step description
    2. Second step description
    3. Final computation
    
    Parameters:
        param1 (type): Description of first parameter. Default: value.
        param2 (type): Description of second parameter. Default: value.
    
    Attributes:
        dependencies (List[str]): Required statistics for computation
        level (str): Estimation level ('sequence', 'token', or 'claim')
        param1: Stored value of param1
        param2: Stored value of param2
    
    Raises:
        ValueError: If parameters are invalid
        TypeError: If wrong types are provided
    
    References:
        Author et al., Year. "Paper Title" Conference/Journal.
        Link: https://arxiv.org/abs/xxxx.xxxxx
    
    Examples:
        Basic usage:
        >>> from lm_polygraph.estimators import YourEstimator
        >>> estimator = YourEstimator(param1=0.5)
        >>> uncertainty = estimate_uncertainty(model, estimator, "Input text")
        
        Advanced usage with custom parameters:
        >>> estimator = YourEstimator(param1=0.8, param2="custom")
        >>> # Use with sampling-based generation
        >>> uncertainty = estimate_uncertainty(
        ...     model, estimator, "Input text",
        ...     generation_params={"num_samples": 10}
        ... )
    
    See Also:
        RelatedEstimator1: Alternative method for similar use case
        RelatedEstimator2: Complementary uncertainty measure
    """
    
    def __init__(self, param1: float = 0.5, param2: str = "default"):
        """
        Initialize the estimator with given parameters.
        
        Parameters:
            param1: Control parameter affecting sensitivity. Range: [0, 1].
            param2: Method variant to use. Options: 'default', 'fast', 'accurate'.
        
        Raises:
            ValueError: If param1 is not in [0, 1] range
        """
        # Determine dependencies based on parameters
        dependencies = ["base_dependency"]
        if param2 == "accurate":
            dependencies.append("additional_dependency")
            
        super().__init__(dependencies, "sequence")
        
        # Validate parameters
        if not 0 <= param1 <= 1:
            raise ValueError(f"param1 must be in [0, 1], got {param1}")
            
        self.param1 = param1
        self.param2 = param2
    
    def __str__(self) -> str:
        """Return unique string identifier including parameters."""
        return f"YourEstimator_p1_{self.param1}_p2_{self.param2}"
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty scores for given statistics.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'base_dependency': Description of what this contains
                - 'additional_dependency': (optional) When param2='accurate'
        
        Returns:
            Array of uncertainty scores. Shape: (n_samples,).
            Higher values indicate higher uncertainty.
            Range: [0, inf) for most cases.
        
        Raises:
            KeyError: If required dependencies are missing from stats
            ValueError: If statistics have unexpected shapes
        """
        # Implementation with clear comments
        base_stats = stats["base_dependency"]
        
        # Step 1: Compute intermediate values
        intermediate = self._compute_intermediate(base_stats)
        
        # Step 2: Apply main transformation
        if self.param2 == "accurate":
            additional = stats["additional_dependency"]
            result = self._accurate_method(intermediate, additional)
        else:
            result = self._default_method(intermediate)
        
        # Step 3: Normalize and return
        return self._normalize(result)
    
    def _compute_intermediate(self, data: np.ndarray) -> np.ndarray:
        """Helper method with its own documentation."""
        # Implementation
        pass
```

## 2. Model Class Method Template

```python
def method_name(
    self,
    input_texts: Union[str, List[str]],
    param1: Optional[int] = None,
    **kwargs
) -> Union[GenerationOutput, List[str]]:
    """
    One-line summary of what this method does.
    
    Detailed explanation including any important behavior or limitations.
    This method is particularly useful for [use case].
    
    Parameters:
        input_texts: Single string or list of strings to process.
            Can also accept List[List[Dict]] for chat-formatted inputs.
        param1: Optional parameter description. If None, uses default.
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature. Default: 1.0
            - max_length (int): Maximum generation length. Default: 100
            - See GenerationParameters for full list
    
    Returns:
        For single input: GenerationOutput object or string
        For batch input: List of outputs
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model generation fails
        
    Examples:
        Single generation:
        >>> output = model.method_name("Hello, world!")
        
        Batch generation:
        >>> outputs = model.method_name(["Text 1", "Text 2"])
        
        With parameters:
        >>> output = model.method_name(
        ...     "Generate text",
        ...     param1=42,
        ...     temperature=0.7,
        ...     max_length=50
        ... )
    
    Note:
        This method requires the model to be in evaluation mode.
        For training, use [alternative method].
    """
```

## 3. Utility Function Template

```python
def utility_function(
    required_arg: np.ndarray,
    optional_arg: Optional[float] = None,
    *args,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    One-line summary of the function purpose.
    
    More detailed explanation of what the function does,
    any algorithms it implements, and when to use it.
    
    Parameters:
        required_arg: Description including expected shape/format
        optional_arg: What this controls. Default: None (auto-computed)
        *args: Additional positional arguments for X purpose
        **kwargs: Keyword arguments:
            - key1 (type): Description
            - key2 (type): Description
    
    Returns:
        Tuple containing:
        - First element: Description with shape
        - Second element: Metadata dictionary with keys:
            - 'info1': Description
            - 'info2': Description
    
    Raises:
        ValueError: Conditions that cause this
        TypeError: Type-related errors
    
    Examples:
        >>> result, metadata = utility_function(
        ...     np.array([1, 2, 3]),
        ...     optional_arg=0.5
        ... )
        >>> print(f"Result shape: {result.shape}")
        >>> print(f"Computation info: {metadata['info1']}")
    
    See Also:
        related_function: Does similar thing differently
    """
```

## 4. Module-level Documentation Template

```python
"""
Module name: Brief description of module purpose.

This module provides [functionality description]. It is primarily used for
[main use case] and supports [key features].

The module contains the following main components:
- Component1: Brief description
- Component2: Brief description
- Component3: Brief description

Typical usage:
    >>> from lm_polygraph.module import MainClass
    >>> instance = MainClass()
    >>> result = instance.process(data)

Key concepts:
    Term1: Definition and importance
    Term2: Definition and importance

Performance considerations:
    - Note about computational complexity
    - Memory usage patterns
    - Optimization tips

Dependencies:
    - External package 1: Why it's needed
    - External package 2: Why it's needed
"""
```

## 5. Configuration Documentation Template

```yaml
# Configuration file for [component name]
# This file controls [what it controls]

# Section 1: Model Configuration
model:
  # Model identifier or path
  # Type: str
  # Default: "gpt2"
  # Example: "meta-llama/Llama-2-7b-hf"
  path: "model_name"
  
  # Device placement strategy
  # Type: str
  # Options: "cuda", "cpu", "auto"
  # Default: "cuda" if available, else "cpu"
  device: "auto"

# Section 2: Generation Parameters
generation:
  # Maximum number of new tokens to generate
  # Type: int
  # Range: [1, model_max_length]
  # Default: 100
  max_new_tokens: 100
  
  # Sampling temperature
  # Type: float
  # Range: (0, inf), typically [0.1, 2.0]
  # Default: 1.0
  # Note: Lower values make output more deterministic
  temperature: 0.8

# Section 3: Uncertainty Estimation
uncertainty:
  # List of estimators to use
  # Type: List[str]
  # Available: See lm_polygraph.estimators.__all__
  # Default: ["Perplexity", "MaximumSequenceProbability"]
  estimators:
    - "SemanticEntropy"
    - "TokenEntropy"
  
  # Estimator-specific configurations
  estimator_configs:
    SemanticEntropy:
      # Number of samples for Monte Carlo estimation
      # Type: int
      # Range: [2, 100], higher is more accurate but slower
      # Default: 10
      num_samples: 20
```

## 6. Error Message Template

```python
class DescriptiveError(Exception):
    """
    Raised when [condition that triggers this error].
    
    This error typically occurs when [common scenario].
    
    Common causes:
        1. Cause one description
        2. Cause two description
        3. Cause three description
    
    How to fix:
        - Solution one
        - Solution two
        - Solution three
    
    Examples:
        This error occurs in the following scenario:
        >>> # Code that would trigger the error
        >>> model = WhiteboxModel(incompatible_model)  # Raises DescriptiveError
    
    See Also:
        RelatedError: For similar but different cases
        Documentation: https://link-to-relevant-docs
    """
    
    def __init__(self, param_name: str, got_value: Any, expected: str):
        """
        Initialize error with helpful context.
        
        Parameters:
            param_name: Name of the problematic parameter
            got_value: The actual value that caused the error
            expected: Description of what was expected
        """
        message = (
            f"Invalid value for '{param_name}': got {got_value!r}, "
            f"expected {expected}.\n"
            f"Hint: Check the documentation for valid values.\n"
            f"Common fix: Ensure the parameter matches one of the expected formats."
        )
        super().__init__(message)
        self.param_name = param_name
        self.got_value = got_value
        self.expected = expected
```

## Usage Guidelines

1. **Consistency**: Use these templates consistently across the codebase
2. **Completeness**: Fill in all sections, remove those that don't apply
3. **Examples**: Always include at least one practical example
4. **Cross-references**: Link related components using "See Also" sections
5. **Updates**: Keep documentation in sync with code changes

## Docstring Checklist

Before committing, ensure your docstrings have:
- [ ] One-line summary
- [ ] Detailed description
- [ ] All parameters documented with types
- [ ] Return value description with type/shape
- [ ] At least one example
- [ ] Raises section if exceptions are possible
- [ ] See Also section if related functionality exists
- [ ] References for academic methods