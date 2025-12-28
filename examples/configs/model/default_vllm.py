from vllm import LLM, SamplingParams


def load_model(
    model_path: str,
    gpu_memory_utilization: float,
    max_new_tokens: int,
    logprobs: int,
    tensor_parallel_size: int = 1,
):
    model = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, logprobs=logprobs)
    return model, sampling_params
