import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric

OPENAI_FACT_CHECK_PROMPT = '''TODO'''


class OpenAIFactCheck(GenerationMetric):
    """
    Calculates for each claim, whether it is true of not, using OpenAI model specified in
    lm_polygraph.stat_calculators.openai_chat.OpenAIChat.
    """

    def __init__(self):
        super().__init__(["input_texts", "openai_chat"], "claim")

    def __str__(self):
        return f"OpenAIFactCheck"

    def _score_single(self, claim: str, input: str, openai_chat) -> bool:
        reply = openai_chat.ask(OPENAI_FACT_CHECK_PROMPT.format(claim=claim, input=input))
        ...

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
            np.ndarray: list of labels, 1 if the fact is true and 0 if not.
        """
        openai_chat = stats["openai_chat"]
        labels = []
        for inp_text, sample_claims in zip(stats["claims"], stats["input_texts"]):
            for claim in sample_claims:
                labels.append(self._score_single(claim.claim_text, inp_text, openai_chat))
        return np.array(labels)
