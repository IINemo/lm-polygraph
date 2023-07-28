import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class PromptCalculator(StatCalculator):
    def __init__(self, prompt: str, expected: str, method: str):
        super().__init__([method], ["greedy_texts", "sample_texts"])
        self.method = method
        self.prompt = prompt
        self.expected = expected

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, **kwargs) -> Dict[str, np.ndarray]:
        expected_tokens = model.tokenizer([self.expected])['input_ids'][0]
        expected_tokens = [t for t in expected_tokens
                           if t != model.tokenizer.eos_token_id and t != model.tokenizer.bos_token_id]
        assert len(expected_tokens) == 1
        expected_token = expected_tokens[0]

        answers = dependencies['greedy_texts']
        samples = dependencies['sample_texts']
        inp_texts = [self.prompt.format(q=text, s=', '.join(sample), a=ans)
                     for text, ans, sample in zip(texts, answers, samples)]

        batch: Dict[str, torch.Tensor] = model.tokenize(inp_texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                min_new_tokens=1,
                max_new_tokens=1,
                num_beams=1,
            )
        logits = torch.stack(out.scores, dim=1).log_softmax(-1)
        log_probs = logits[:, -1, expected_token].cpu().numpy()

        return {self.method: log_probs}
