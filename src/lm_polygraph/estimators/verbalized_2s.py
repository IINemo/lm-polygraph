import numpy as np
import re

from typing import Dict

from .estimator import Estimator


class Verbalized2S(Estimator):
    """
    Asks model to output it's confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """

    def __init__(
        self,
        confidence_prompt: str,
        confidence_regex: str = "",
        max_new_tokens: int = 10,
        name_postfix="",
    ):
        self.max_new_tokens = max_new_tokens
        self.confidence_prompt = confidence_prompt
        self.confidence_regex = confidence_regex
        self.postfix = name_postfix
        super().__init__(["input_texts", "greedy_texts"], "sequence")

    def __str__(self):
        return f"Verbalized2S{self.postfix}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        chats = []
        prompts = stats["input_texts"]
        guesses = stats["greedy_texts"]
        for prompt, guess in zip(prompts, guesses):
            chats.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": guess},
                    {"role": "user", "content": self.confidence_prompt},
                ]
            )

        out = model.generate_texts(
            chats,
            min_new_tokens=2,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
        )

        ues = []
        conf_re = re.compile(self.confidence_regex)
        for answer in out:
            match = re.search(conf_re, answer)

            try:
                ue = 1 - float(match.groups()[0])
            except AttributeError:
                ue = np.nan

            ues.append(ue)

        return np.array(ues)
