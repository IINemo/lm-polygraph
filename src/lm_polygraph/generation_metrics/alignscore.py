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
        lang="en",
        ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
        batch_size=16,
        target_is_claims=True,
        ignore_target=False,
        sample: bool = False,
    ):
        if sample:
            super().__init__(["first_sample_texts", "input_texts"], "sequence")
        else:
            super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.sample = sample
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_is_claims = target_is_claims
        self.batch_size = batch_size
        self.ignore_target = ignore_target
        self.scorer = AlignScorer(
            model="roberta-large",
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )

    def __str__(self):
        base = "AlignScore"
        if self.ignore_target:
            base += "InputOutput"
        elif self.target_is_claims:
            base += "OutputTarget"
        else:
            base += "TargetOutput"

        if self.sample:
            return f"Sample{base}"

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
            gen_texts = stats["first_sample_texts"]
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
