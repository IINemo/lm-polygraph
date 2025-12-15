from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map: str, token: str = None, max_memory: dict = None):
    """
    Load a causal language model from Hugging Face.
    
    Parameters:
        model_path: Path to the model on Hugging Face
        device_map: Device mapping (e.g., "auto", "cuda:0")
        token: Hugging Face authentication token (for gated models). 
               If None, will try environment variables or huggingface_hub login cache.
        max_memory: Maximum memory to use per device (e.g., {0: "40GiB", 1: "40GiB"}).
                    If None and device_map is "auto", will use default (90% of GPU memory).
                    Recommended: Use 70-80% to leave room for long inputs with tool-enhanced prompts.
    """
    # Get token from parameter or environment variable
    # If still None, transformers will use huggingface_hub login cache automatically
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    # Build kwargs - only include token if explicitly provided
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if token is not None:
        load_kwargs["token"] = token
    
    # Add max_memory if provided (useful for limiting GPU memory usage)
    # When using tool calling with long prompts, reduce max_memory to 70-75% to avoid OOM
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs
    )
    model.eval()
    
    # Set pad_token_id in generation_config to suppress warnings
    # This is set after model load to ensure tokenizer is available
    # The pad_token_id will be set when tokenizer is loaded, but we set it here
    # to prevent the warning during generation
    if hasattr(model, 'generation_config') and model.generation_config.pad_token_id is None:
        if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
            model.generation_config.pad_token_id = model.config.eos_token_id

    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True, token: str = None):
    """
    Load a tokenizer from Hugging Face.
    
    Parameters:
        model_path: Path to the model on Hugging Face
        add_bos_token: Whether to add BOS token
        token: Hugging Face authentication token (for gated models).
               If None, will try environment variables or huggingface_hub login cache.
    """
    # Get token from parameter or environment variable
    # If still None, transformers will use huggingface_hub login cache automatically
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    # Build kwargs - only include token if explicitly provided
    load_kwargs = {
        "padding_side": "left",
        "add_bos_token": add_bos_token,
    }
    if token is not None:
        load_kwargs["token"] = token
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        **load_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
