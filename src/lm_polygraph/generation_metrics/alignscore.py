import numpy as np
from .alignscore_utils import AlignScorer

import torch
from typing import List, Dict
from .generation_metric import GenerationMetric


class AlignScore(GenerationMetric):
    """
    Calculates AlignScore metric (https://aclanthology.org/2023.acl-long.634/)
    between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        scorer,
        lang="en",
        target_is_claims=True,
        ignore_target=False,
        sample: bool = False,
        sample_strategy: str = "First",
    ):
        if sample:
            super().__init__([
                "first_sample_texts",
                "best_sample_texts",
                "best_normalized_sample_texts",
                "mbr_sample_texts",
                "input_texts"],
            "sequence")
        else:
            super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.sample = sample
        self.sample_strategy = sample_strategy
        self.target_is_claims = target_is_claims
        self.ignore_target = ignore_target
        self.scorer = scorer

    def __str__(self):
        base = "AlignScore"
        if self.ignore_target:
            base += "InputOutput"
        elif self.target_is_claims:
            base += "OutputTarget"
        else:
            base += "TargetOutput"

        if self.sample:
            if self.sample_strategy == "First":
                return f"Sample{base}"
            else:
                return f"{self.sample_strategy}Sample{base}"

        return base

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates AlignScore (https://aclanthology.org/2023.acl-long.634/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of AlignScore Scores for each sample in input.
        """
        if self.sample:
            if self.sample_strategy == "First":
                gen_texts = stats["first_sample_texts"]
            elif self.sample_strategy == "Best":
                gen_texts = stats["best_sample_texts"]
            elif self.sample_strategy == "BestNormalized":
                gen_texts = stats["best_normalized_sample_texts"]
            elif self.sample_strategy == "Mbr":
                gen_texts = stats["mbr_sample_texts"]
            else:
                raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")
        else:
            gen_texts = stats["greedy_texts"]

        input_texts = stats["input_texts"]

        filtered_targets = [x if len(x.strip()) else "(empty)" for x in target_texts]
        filtered_outputs = [x if len(x.strip()) else "(empty)" for x in gen_texts]
        filtered_inputs = [x if len(x.strip()) else "(empty)" for x in input_texts]

        if self.ignore_target:
            claims = filtered_outputs
            contexts = filtered_inputs
        else:
            if self.target_is_claims:
                claims = filtered_targets
                contexts = filtered_outputs
            else:
                claims = filtered_outputs
                contexts = filtered_targets

        scores = np.array(
            self.scorer.score(
                claims=claims,
                contexts=contexts,
            )
        )

        return scores
