from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.stat_calculators import InferCausalLMCalculator, EntropyCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty

import torch


def test_CausalLMWithUncertainty():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(device)

    stat_calculators = [InferCausalLMCalculator(tokenize=False), EntropyCalculator()]
    estimator = MeanTokenEntropy()
    llm_with_uncertainty = CausalLMWithUncertainty(
        llm, tokenizer, stat_calculators, estimator
    )

    prompts = ["Write a short story about a robot learning to paint.\n"]

    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    chat_prompts = tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(chat_prompts, return_tensors="pt").to(device)

    output = llm_with_uncertainty.generate(
        **inputs, max_new_tokens=30, temperature=0.7, do_sample=True
    )

    print("LLM output:")
    print(tokenizer.decode(output.sequences[0], skip_special_tokens=True))
    print("Uncertainty score: ", output.uncertainty_score)
