import numpy as np

from collections import defaultdict
from typing import List, Dict, Optional

from .estimator import Estimator


class SemanticEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(
            [
                "sample_log_probs",
                "sample_texts",
                "semantic_classes_entail",
            ],
            "sequence",
        )
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]

        self._class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        self._sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        # Concatenate hypos with input texts
        hyps_list = [[] for _ in stats["input_texts"]]
        for i, input_text in enumerate(stats["input_texts"]):
            for hyp in stats["sample_texts"][i]:
                hyps_list[i].append(" ".join([input_text, hyp]))

        return self.batched_call(hyps_list, loglikelihoods_list)

    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        log_weights: Optional[List[List[float]]] = None,
    ) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_sample[i]
            ]
            class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods
            ]
            if log_weights[i] is None:
                log_weights[i] = [0 for _ in hyps_list[i]]
            semantic_logits[i] = -np.mean(
                [
                    class_lp[self._sample_to_class[i][j]] * np.exp(log_weights[i][j])
                    for j in range(len(hyps_list[i]))
                ]
            )
        return np.array([semantic_logits[i] for i in range(len(hyps_list))])
