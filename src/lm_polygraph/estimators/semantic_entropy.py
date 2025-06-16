import numpy as np

from typing import Dict

from .estimator import Estimator
from .utils.semantic_entropy import compute_semantic_entropy


class SemanticEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
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
            deps = ["sample_log_probs", "sample_texts", "semantic_classes_entail"]
        elif self.class_probability_estimation == "frequency":
            deps = ["sample_texts", "semantic_classes_entail"]
        else:
            raise ValueError(
                f"Unknown class_probability_estimation: {self.class_probability_estimation}. Use 'sum' or 'frequency'."
            )

        super().__init__(deps, "sequence")
        self.verbose = verbose

    def __str__(self):
        base = "SemanticEntropy"
        if self.class_probability_estimation == "frequency":
            base += "Empirical"
        if self.entropy_estimator == "direct":
            base = "Direct" + base
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * semantic class mapping in 'semantic_classes_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        if self.class_probability_estimation == "sum":
            loglikelihoods_list = stats["sample_log_probs"]
            hyps_list = stats["sample_texts"]
        else:
            loglikelihoods_list = None
            hyps_list = stats["sample_texts"]

        class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        return compute_semantic_entropy(
            hyps_list,
            loglikelihoods_list,
            class_to_sample,
            sample_to_class,
            self.class_probability_estimation,
            self.entropy_estimator,
        )
