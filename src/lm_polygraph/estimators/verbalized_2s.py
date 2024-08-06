import numpy as np
import re

from typing import Dict

from .estimator import Estimator


VERB_2S_FIRST_PROMPT_COT = "Provide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is: {q}"
VERB_2S_SECOND_PROMPT_COT = "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n"

VERB_2S_FIRST_PROMPT_TOP1 = "Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is:{q}"
VERB_2S_SECOND_PROMPT_TOP1 = VERB_2S_SECOND_PROMPT_COT

VERB_2S_FIRST_PROMPT_TOPK = "Provide your {k} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!> ... G{k}: <{k}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is:{q}"
VERB_2S_SECOND_PROMPT_TOPK = "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\nFor example:\n\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n... P{k}: <the probability between 0.0 and 1.0 that G{k} is correct, without any extra commentary whatsoever; just the probability!>"

TOP1_CONFIDENCE_REGEX = r"(\d+\.\d+)"
TOPK_CONFIDENCE_REGEX = r"P1: (\d+\.\d+)"


class Verbalized2S(Estimator):
    def __init__(self, topk=1, cot=False):
        self.cot = cot
        if cot:
            self.topk = 1
            super().__init__([], "sequence")
        else:
            self.topk = topk
            super().__init__([], "sequence")

    def __str__(self):
        if self.cot:
            return f"Verbalized2SCoT"
        return f"Verbalized2STop{self.topk}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        first_prompts = []
        second_prompts = []
        for text in stats["input_texts"]:
            if self.cot:
                first_prompt = VERB_2S_FIRST_PROMPT_COT.format(q=text)
                second_prompt = VERB_2S_SECOND_PROMPT_COT
            if self.topk > 1:
                first_prompt = VERB_2S_FIRST_PROMPT_TOPK.format(k=self.topk, q=text)
                second_prompt = VERB_2S_SECOND_PROMPT_TOPK.format(k=self.topk)
            else:
                first_prompt = VERB_2S_FIRST_PROMPT_TOP1.format(q=text)
                second_prompt = VERB_2S_SECOND_PROMPT_TOP1
            first_prompts.append(first_prompt)
            second_prompts.append(second_prompt)

        max_new_tokens = self.topk * stats["max_new_tokens"]
        guesses = model.generate_texts(
            first_prompts,
            min_new_tokens=2,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )

        chats = []
        for first_prompt, guess, second_prompt in zip(first_prompts, guesses, second_prompts):
            chats.append([
                {'role': 'user', 'content': first_prompt},
                {'role': 'assistant', 'content': guess},
                {'role': 'user', 'content': second_prompt},
            ])

        # 10 tokens for confidence, times number of guesses
        max_new_tokens = self.topk * 10

        out = model.generate_texts(
            chats,
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
