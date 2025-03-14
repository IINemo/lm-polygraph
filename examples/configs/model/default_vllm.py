from vllm import LLM, SamplingParams


def load_model(
    model_path: str, gpu_memory_utilization: float, max_new_tokens: int, logprobs: int
):
    model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, logprobs=logprobs)
    return model, sampling_params
