import numpy as np
import os

from typing import List, Dict
from lm_polygraph.utils.openai_chat import OpenAIChat
from .generation_metric import GenerationMetric

OPENAI_FACT_CHECK_PROMPT = (
    "Is the claim correct according to the most "
    "recent sources of information? "
    """Answer "True", "False" or "Not known"."""
    "\n\n"
    "Examples:\n"
    "\n"
    "Question: Tell me a bio of Albert Einstein.\n"
    "Claim: He was born on 14 March.\n"
    "Answer: True\n"
    "\n"
    "Question: Tell me a bio of Albert Einstein.\n"
    "Claim: He was born in United Kingdom.\n"
    "Answer: False\n"
    "\n"
    "Your input:\n"
    "\n"
    "Question: {input}\n"
    "Claim: {claim}\n"
    "Answer: "
)


class OpenAIFactCheck(GenerationMetric):
    """
    Calculates for each claim, whether it is true of not, using OpenAI model specified in
    lm_polygraph.stat_calculators.openai_chat.OpenAIChat.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4",
        cache_path: str = os.path.expanduser("~") + "/.cache",
    ):
        super().__init__(["input_texts"], "claim")
        self.openai_chat = OpenAIChat(openai_model=openai_model, cache_path=cache_path)

    def __str__(self):
        return "OpenAIFactCheck"

    def _score_single(self, claim: str, input: str, openai_chat) -> int:
        reply = openai_chat.ask(
            OPENAI_FACT_CHECK_PROMPT.format(
                claim=claim,
                input=input,
            )
        )
        reply = reply.strip()
        if reply == "True":
            return 0
        elif reply == "False":
            return 1
        else:
            return np.nan

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        For each claim in stats['claims'], asks OpenAI model whether this fact is correct or not.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * for each generation, list of lm_polygraph.stat_calculators.extract_claims.Claim in 'claims'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of labels, 1 if the fact is false and 0 if it is true.
        """
        labels = []
        for inp_text, sample_claims in zip(stats["input_texts"], stats["claims"]):
            labels.append([])
            for claim in sample_claims:
                labels[-1].append(
                    self._score_single(
                        claim.claim_text,
                        inp_text,
                        self.openai_chat,
                    )
                )
        return labels
