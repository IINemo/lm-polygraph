import numpy as np
import torch

from typing import Dict

from .estimator import Estimator


class SemanticEntropySDLG(Estimator):
    """
    Estimates the sequence-level uncertainty using semantic entropy with SDLG sampling.
    Reuses the core semantic entropy logic but based on SDLG sampler.
    """

    def __init__(self):
        deps = [
            "sdlg_sample_likelihoods",
            "sdlg_sample_texts",
            "sdlg_semantic_classes_entail",
            "sdlg_sample_importance_weights",
        ]
        super().__init__(deps, "sequence")

    def __str__(self):
        return "SemanticEntropySDLG"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample using SDLG sampling.
        Implementation according to paper https://arxiv.org/pdf/2406.04306.

        Parameters:
            stats: dictionary containing:
                * sdlg_sample_texts: generated SDLG samples
                * sdlg_sample_likelihoods: log-probabilities for each SDLG sample (if sum estimation)
                * sdlg_semantic_classes_entail: semantic class mappings with keys 'sample_to_class' and 'class_to_sample'

        Returns:
            np.ndarray: float semantic entropy for each sample. Higher means more uncertain.
        """
        loglikelihoods_list = stats["sdlg_sample_likelihoods"]
        hyps_list = stats["sdlg_sample_texts"]
        weights = stats["sdlg_sample_importance_weights"]

        class_to_sample = stats["sdlg_semantic_classes_entail"]["class_to_sample"]
        sample_to_class = stats["sdlg_semantic_classes_entail"]["sample_to_class"]

        return self.compute_semantic_entropy(
            hyps_list,
            loglikelihoods_list,
            class_to_sample,
            sample_to_class,
            weights=weights,
        )

    def compute_semantic_entropy(
        self,
        hyps_list,
        loglikelihoods_list,
        class_to_sample,
        sample_to_class,
        weights,
    ):
        """
        Computes SDLG-specific semantic entropy for sequence-level uncertainty.
        Args:
            hyps_list: List of generation samples per input.
            loglikelihoods_list: List of log-probabilities per sample, or None if using frequency.
            class_to_sample: Mapping from class indices to sample indices.
            sample_to_class: Mapping from sample indices to class indices.

        Returns:
            np.ndarray of semantic entropy values per input.
        """
        results = []
        for i, hyps in enumerate(hyps_list):
            likelihoods = loglikelihoods_list[i]
            weights = np.array(weights[i])

            # Importance sampling adjustment
            class_likelihoods = [
                np.array(likelihoods)[np.array(cls)] + np.log(weights[np.array(cls)])
                for cls in class_to_sample[i]
            ]
            class_lp = [
                np.logaddexp.reduce(likelihood) for likelihood in class_likelihoods
            ]

            # This is incorrect. We multiply class log probs from unnormalized distribution
            # with normalized class probabilities. But this is the original implementation
            # that comes with the paper.
            class_p = torch.softmax(torch.tensor(class_lp), dim=0).numpy()
            ent = -np.sum(
                [
                    class_lp[sample_to_class[i][j]] * class_p[sample_to_class[i][j]]
                    for j in range(len(hyps))
                ]
            )

            results.append(ent)

        return np.array(results)
