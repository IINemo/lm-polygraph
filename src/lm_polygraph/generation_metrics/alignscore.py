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
    ):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = AlignScorer(
            model="roberta-large",
            batch_size=16,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )

    def __str__(self):
        return "AlignScore"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates AlignScore (https://aclanthology.org/2023.acl-long.634/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of AlignScore Scores for each sample in input.
        """
        scores = np.array(
            self.scorer.score(
                claims=stats["target_texts"],
                contexts=[x if len(x) else "-" for x in stats["greedy_texts"]],
            )
        )
        return scores
