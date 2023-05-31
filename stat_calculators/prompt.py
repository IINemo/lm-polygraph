import torch
import numpy as np

from typing import Dict, List

from stat_calculators.stat_calculator import StatCalculator
from utils.model import Model


class PromptCalculator(StatCalculator):
    def __init__(self, prompt: str, expected: str, method: str):
        super().__init__([method], ['greedy_texts'])
        self.method = method
        self.prompt = prompt
        self.expected = expected

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        expected_tokens = model.tokenizer([self.expected])['input_ids'][0]
        assert len(expected_tokens) == 1
        expected_token = expected_tokens[0]

        answers = dependencies['greedy_texts']
        inp_texts = [self.prompt.format(q=text, a=ans) for text, ans in zip(texts, answers)]
        batch: Dict[str, torch.Tensor] = model.tokenize(inp_texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1,
            )
        logits = torch.stack(out.scores, dim=1).log_softmax(-1)
        log_probs = logits[:, -1, expected_token].cpu().numpy()

        return {self.method: log_probs}
