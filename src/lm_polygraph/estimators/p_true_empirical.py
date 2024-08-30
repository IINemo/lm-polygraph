import numpy as np

from typing import Dict

from .estimator import Estimator


class PTrueEmpirical(Estimator):
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        super().__init__(["input_texts", "greedy_texts"], "sequence")

    def __str__(self):
        return "PTrueEmpirical"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        prompts = stats["input_texts"]
        guesses = stats["greedy_texts"]
        model = stats["model"]

        ues = []
        for prompt, guess in zip(prompts, guesses):
            input = f"Question:\n\n{prompt}\n\nProposed Answer: {guess}\n\nIs the proposed answer:\n\tTrue or\n\tFalse?\nThe proposed answer is: "

            out = model.generate_texts(
                [input],
                min_new_tokens=1,
                max_new_tokens=1,
                num_return_sequences=self.num_samples,
                do_sample=True,
            )

            ue = 1 - np.mean([1 if "True" in text else 0 for text in out])
            ues.append(ue)

        return np.array(ues)
