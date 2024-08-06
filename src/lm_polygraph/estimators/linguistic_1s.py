import numpy as np
import re

from typing import Dict

from .estimator import Estimator

EXPRESSION_LIST = {
    "Almost Certain": 0.95,
    "Highly Likely": 0.9,
    "Very Good Chance": 0.8,
    "We Beleive": 0.75,
    "Probably": 0.7,
    "Probable": 0.7,
    "Likely": 0.7,
    "Better than Even": 0.6,
    "About Even": 0.5,
    "Probably Not": 0.25,
    "We Doubt": 0.2,
    "Unlikely": 0.2,
    "Little Chance": 0.1,
    "Chances Are Slight": 0.1,
    "Improbable": 0.1,
    "Highly Unlikely": 0.05,
    "Almost No Chance": 0.02,
}

LING_1S_PROMPT = "Provide your best guess for the following question, and describe how likely it is that your guess is correct as one of the following expressions:\n{e}.\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe question is: {q}"


class Linguistic1S(Estimator):
    def __init__(self):
        super().__init__([], "sequence")

    def __str__(self):
        return f"Linguistic1S"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        prompts = []
        expressions = "\n".join(EXPRESSION_LIST.keys())
        for text in stats["input_texts"]:
            prompt = LING_1S_PROMPT.format(q=text, e=expressions)
            prompts.append(prompt)

        # 10 tokens for confidence, times number of guesses
        max_new_tokens=stats["max_new_tokens"] + 20

        out = model.generate_texts(
            prompts,
            min_new_tokens=2,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )

        ues = []
        for answer in out:
            ue = np.nan
            for expression, confidence in EXPRESSION_LIST.items():
                if expression in answer:
                    ue = 1 - confidence
                    break

            ues.append(ue)

        return np.array(ues)
