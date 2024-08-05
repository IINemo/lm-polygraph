import numpy as np

from typing import Dict

from .estimator import Estimator


VERB_1S_PROMPT_TOP1 = "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {q}"
VERB_1S_PROMPT_TOPK = "Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!> ... G{k}: <{k}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nP{k}: <the probability between 0.0 and 1.0 that G{k} is correct, without any extra commentary whatsoever; just the probability!> \n\nThe question is: {q}"

TOP1_CONFIDENCE_REGEX = r"\d+\.\d+"
TOPK_CONFIDENCE_REGEX = r"P1: (\d+\.\d+)"


class Verbalized1S(Estimator):
    def __init__(self, topk=1):
        self.topk = topk
        super().__init__([], "sequence")

    def __str__(self):
        return f"Verbalized1STop{self.topk}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        prompts = []
        for text in stats["input_texts"]:
            if self.topk > 1:
                prompt = VERB_1S_PROMPT_TOPK.format(k=self.topk, q=text)
            else:
                prompt = VERB_1S_PROMPT_TOP1.format(q=text)
            prompts.append(prompt)

        # 10 tokens for confidence, times number of guesses
        max_new_tokens=self.topk * (stats["max_new_tokens"] + 10) 

        out = model.generate_texts(
            prompts,
            min_new_tokens=2,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )

        ues = []
        conf_re = TOPK_CONFIDENCE_REGEX if self.topk > 1 else TOP1_CONFIDENCE_REGEX
        for answer in out:
            match = re.search(conf_re, answer)

            try:
                ue = 1 - float(match.groups()[0])
            except AttributeError:
                ue = np.nan

            ues.append(ue)

        return np.array(ues)
