import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class GreedyLMProbsCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['greedy_lm_log_probs', 'greedy_lm_log_likelihoods', 'train_greedy_lm_log_likelihoods'], ['greedy_tokens'])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, max_new_tokens: int = 100, **kwargs) -> Dict[str, np.ndarray]:
        tokens = dependencies['greedy_tokens']
        try:
            batch = model.tokenize([model.tokenizer.decode(t) for t in tokens])
            batch = {k: v.to(model.device()) for k, v in batch.items()}
            with torch.no_grad():
                if model.model_type == "Seq2SeqLM":
                    logprobs = model.model(**batch, decoder_input_ids=batch["input_ids"]).logits.log_softmax(-1)
                else:
                    logprobs = model.model(**batch).logits.log_softmax(-1)
            greedy_lm_log_probs = []
            greedy_lm_ll = []
            for i in range(len(tokens)):
                assert len(logprobs[i]) >= len(tokens[i])
                greedy_lm_log_probs.append(logprobs[i, -len(tokens[i]):-1].cpu().numpy())
                greedy_lm_ll.append([logprobs[i, -len(tokens[i]) + j, tokens[i][j]].item()
                                     for j in range(len(tokens[i]))])
        except:
            # case where tokenizer(tokenizer.decode(t)) != t; process each sequence separetly
            greedy_lm_log_probs = []
            greedy_lm_ll = []
            for toks in tokens:
                input_ids = torch.LongTensor([toks]).to(model.device())
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids).to(model.device())
                }
                with torch.no_grad():
                    if model.model_type == "Seq2SeqLM":
                        logprobs = model.model(**batch, decoder_input_ids=batch["input_ids"]).logits.log_softmax(-1)
                    else:
                        logprobs = model.model(**batch).logits.log_softmax(-1)
                logprobs = logprobs[0]
                assert len(logprobs) >= len(toks)
                greedy_lm_log_probs.append(logprobs[-len(toks):-1].cpu().numpy())
                greedy_lm_ll.append([logprobs[-len(toks) + j, toks[j]].item()
                                     for j in range(len(toks))])
        return {
            'greedy_lm_log_probs': greedy_lm_log_probs,
            'greedy_lm_log_likelihoods': greedy_lm_ll,
        }
