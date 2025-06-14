import numpy as np

from typing import Dict

from .estimator import Estimator
from .utils.semantic_entropy import compute_semantic_entropy


class SemanticEntropySDLG(Estimator):
    """
    Estimates the sequence-level uncertainty using semantic entropy with SDLG sampling.
    Reuses the core semantic entropy logic but based on SDLG sampler.
    """

    def __init__(
        self,
        verbose: bool = False,
        class_probability_estimation: str = "sum",
        entropy_estimator: str = "mean",
    ):
        self.class_probability_estimation = class_probability_estimation
        self.entropy_estimator = entropy_estimator
        if self.class_probability_estimation == "sum":
            deps = [
                "sdlg_sample_likelihoods",
                "sdlg_sample_texts",
                "sdlg_semantic_classes_entail",
            ]
        elif self.class_probability_estimation == "frequency":
            deps = ["sdlg_sample_texts", "sdlg_semantic_classes_entail"]
        else:
            raise ValueError(
                f"Unknown class_probability_estimation: {self.class_probability_estimation}. "
                f"Use 'sum' or 'frequency'."
            )

        super().__init__(deps, "sequence")
        self.verbose = verbose

    def __str__(self):
        base = "SemanticEntropySDLG"
        if self.class_probability_estimation == "frequency":
            base += "Empirical"
        if self.entropy_estimator == "direct":
            base = "Direct" + base
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample using SDLG sampling.

        Parameters:
            stats: dictionary containing:
                * sdlg_sample_texts: generated SDLG samples
                * sdlg_sample_likelihoods: log-probabilities for each SDLG sample (if sum estimation)
                * sdlg_semantic_classes_entail: semantic class mappings with keys 'sample_to_class' and 'class_to_sample'

        Returns:
            np.ndarray: float semantic entropy for each sample. Higher means more uncertain.
        """
        if self.class_probability_estimation == "sum":
            loglikelihoods_list = stats["sdlg_sample_likelihoods"]
            hyps_list = stats["sdlg_sample_texts"]
        else:
            loglikelihoods_list = None
            hyps_list = stats["sdlg_sample_texts"]

        class_to_sample = stats["sdlg_semantic_classes_entail"]["class_to_sample"]
        sample_to_class = stats["sdlg_semantic_classes_entail"]["sample_to_class"]

        return compute_semantic_entropy(
            hyps_list,
            loglikelihoods_list,
            class_to_sample,
            sample_to_class,
            self.class_probability_estimation,
            self.entropy_estimator,
        )
