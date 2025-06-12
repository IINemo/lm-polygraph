import numpy as np

from typing import Dict

from .estimator import Estimator


class SemanticEntropySDLG(Estimator):
    """
        This estimator implements Semantic Entropy variation as defined in the paper
        "Semantically Diverse Language Generation..." by Aichberger et al. (2024).
        https://arxiv.org/pdf/2406.04306
    """

    def __init__(self):
        super().__init__(
            [
                "sdlg_sample_likelihoods",
                "sdlg_sample_texts",
                "sdlg_sample_tokens",
            ],
            "sequence"
        )

    def __str__(self):
        return "SemanticEntropySDLG"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        pass
