import torch
import numpy as np

from typing import Dict, List, Optional

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class PromptCalculator(StatCalculator):
    """
    Calculates the probability for a specific token to be generated from the specific prompt.
    Used for P(True)-based methods.
    """

    def __init__(
        self,
        prompt: str,
        expected: str,
        method: str,
        input_text_dependency: str = "input_texts",
        sample_text_dependency: Optional[str] = "sample_texts",
        generation_text_dependency: str = "greedy_texts",
    ):
        """
        Parameters:
            prompt (str): Prompt to use for estimating the answer of.
                The following values can be used in the prompt:
                - q: input text
                - a: generation text
                - s: list of several generation samples.
                Prompt example: 'Question: {q}. Is the following answer true? {a}'.
            expected (str): string to measure probability of. Must be decoded into one token,
                otherwise an exception will be raised.
            method (str): the name of the statistics to calculate with this calculator.
        """
        dependencies = [input_text_dependency, generation_text_dependency]
        if "{s}" in prompt and sample_text_dependency is not None:
            dependencies.append(sample_text_dependency)
        super().__init__([method], dependencies)
        self.method = method
        self.prompt = prompt
        self.expected = expected
        self.input_text_dependency = input_text_dependency
        self.sample_text_dependency = sample_text_dependency
        self.generation_text_dependency = generation_text_dependency

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the probability for `expected` to be generated from `prompt`.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_texts' (List[str]): model generations for this batch,
                - 'sample_texts' (List[List[str]]): several sampling generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - `method` (List[float]): logarithms of probability of generating `expected` from prompt formatted
                    at each input text.
        """
        expected_tokens = model.tokenizer([self.expected])["input_ids"][0]
        expected_tokens = [
            t
            for t in expected_tokens
            if t != model.tokenizer.eos_token_id and t != model.tokenizer.bos_token_id
        ]
        assert len(expected_tokens) == 1
        expected_token = expected_tokens[0]

        answers = dependencies[self.generation_text_dependency]
        samples = [[] for _ in range(len(answers))]
        if self.sample_text_dependency is not None:
            samples = dependencies[self.sample_text_dependency]
        input_texts = dependencies[self.input_text_dependency]
        inp_texts = [
            self.prompt.format(q=text, s=", ".join(sample), a=ans)
            for text, ans, sample in zip(input_texts, answers, samples)
        ]

        if len(inp_texts) == 0:
            return {self.method: np.array([])}

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

        logits = torch.stack(out.scores, dim=1)
        log_probs = logits[:, -1, expected_token].cpu().numpy()

        return {self.method: log_probs}
