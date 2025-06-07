import numpy as np
import os
from tqdm import tqdm

from typing import List, Dict
from lm_polygraph.utils.openai_chat import OpenAIChat
from .generation_metric import GenerationMetric
from lm_polygraph.stat_calculators.claim_level_prompts import *

from concurrent.futures import ThreadPoolExecutor


class OpenAIFactCheck(GenerationMetric):
    """
    Calculates for each claim, whether it is true of not, using OpenAI model specified in
    lm_polygraph.stat_calculators.openai_chat.OpenAIChat.
    """

    def __init__(
        self,
        llm_url: str = None,
        openai_model: str = "gpt-4o",
        cache_path: str = os.path.expanduser("~") + "/.cache",
        language: str = "en",
        progress_bar: bool = False,
        fact_check_prompts: Dict[str, str] = OPENAI_FACT_CHECK_PROMPTS,
        fact_check_summarize_prompt: Dict[
            str, str
        ] = OPENAI_FACT_CHECK_SUMMARIZE_PROMPT,
        n_threads: int = 1,
        timeout: int = 600,
        max_tokens: int = None,
        rewrite_cache: bool = False,
    ):
        super().__init__(["input_texts"], "claim")
        self.openai_chat = OpenAIChat(
            base_url=llm_url,
            openai_model=openai_model,
            cache_path=cache_path,
            timeout=timeout,
            max_tokens=max_tokens,
            rewrite_cache=rewrite_cache,
        )
        self.language = language
        self.progress_bar = progress_bar
        self.fact_check_prompts = fact_check_prompts
        self.fact_check_summarize_prompt = fact_check_summarize_prompt
        self.n_threads = n_threads

    def __str__(self):
        return "OpenAIFactCheck"

    def _score_single(self, args: tuple[str, str]) -> int:
        claim, input = args
        reply = self.openai_chat.ask(
            self.fact_check_prompts[self.language].format(
                claim=claim,
                input=input,
            )
        )
        reply = self.openai_chat.ask(
            self.fact_check_summarize_prompt[self.language].format(
                claim=claim,
                input=input,
                reply=reply,
            )
        )
        reply = reply.strip()
        if any(x in reply for x in ["True", '"True"', "是", "真", "نعم"]):
            return 0
        elif any(x in reply for x in ["False", '"False"', "否", "假", "لا"]):
            return 1
        else:
            return np.nan

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        For each claim in stats['claims'], asks OpenAI model whether this fact is correct or not.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * for each generation, list of lm_polygraph.stat_calculators.extract_claims.Claim in 'claims'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of labels, 1 if the fact is false and 0 if it is true.
        """
        input_texts = stats["input_texts"]

        all_inputs = [
            (claim.claim_text, input_text)
            for input_text, sample_claims in zip(input_texts, stats["claims"])
            for claim in sample_claims
        ]

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            all_outputs = list(
                tqdm(
                    executor.map(self._score_single, all_inputs),
                    total=len(all_inputs),
                    desc="Verifying claims",
                    disable=not self.progress_bar,
                )
            )

        claim_labels = []
        for sample_claims in stats["claims"]:
            claim_labels.append(all_outputs[: len(sample_claims)])
            all_outputs = all_outputs[len(sample_claims) :]

        return claim_labels
