import numpy as np

from collections import defaultdict
from typing import List, Dict, Optional

from .estimator import Estimator


class SemanticEntropy(Estimator):
    """ Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(
        self,
        verbose: bool = False,
        use_unique_responses: bool = False,
        mode: str = "output"
    ):
        if mode == 'output':
            super().__init__(
                [
                    "sample_log_probs",
                    "sample_texts",
                    "semantic_matrix_entail",
                    "entailment_id",
                ],
                "sequence",
            )
        elif mode == 'input_output':
            super().__init__(
                [
                    "sample_log_probs",
                    "sample_texts",
                    "concat_semantic_matrix_entail",
                    "entailment_id",
                ],
                "sequence",
            )
        self.mode = mode
        self.use_unique_responses = use_unique_responses
        self.verbose = verbose

    def __str__(self):
        base = "SemanticEntropy"
        if self.mode == 'input_output':
            base += "Concat"
        if self.use_unique_responses:
            base += "Unique"
        return base

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
        hyps_list = stats["sample_texts"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        if self.mode == 'output':
            self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]
        elif self.mode == 'input_output':
            self._is_entailment = stats["concat_semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(hyps_list, loglikelihoods_list)

    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        log_weights: Optional[List[List[float]]] = None,
    ) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_sample[i]
            ]
            if self.use_unique_responses:
                class_hyps = [
                    np.array(hyps_list[i])[np.array(class_idx)]
                    for class_idx in self._class_to_sample[i]
                ]
                unique_hyps_ids = [
                    np.unique(hyps, return_index=True)[1]
                    for hyps in class_hyps
                ]
                class_likelihoods = [
                    likelihoods[ids]
                    for ids, likelihoods in zip(unique_hyps_ids, class_likelihoods)
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

    def get_classes(self, hyps_list: List[List[str]]):
        self._sample_to_class = {}
        self._class_to_sample: Dict[int, List] = defaultdict(list)

        [
            self._determine_class(idx, i)
            for idx, hyp in enumerate(hyps_list)
            for i in range(len(hyp))
        ]

        return self._sample_to_class, self._class_to_sample

    def _determine_class(self, idx: int, i: int):
        # For first hypo just create a zeroth class
        if i == 0:
            self._class_to_sample[idx] = [[0]]
            self._sample_to_class[idx] = {0: 0}

            return 0

        # Iterate over existing classes and return if hypo belongs to one of them
        for class_id in range(len(self._class_to_sample[idx])):
            class_text_id = self._class_to_sample[idx][class_id][0]
            forward_entailment = self._is_entailment[idx, class_text_id, i]
            backward_entailment = self._is_entailment[idx, i, class_text_id]
            if forward_entailment and backward_entailment:
                self._class_to_sample[idx][class_id].append(i)
                self._sample_to_class[idx][i] = class_id

                return class_id

        # If none of the existing classes satisfy - create new one
        new_class_id = len(self._class_to_sample[idx])
        self._sample_to_class[idx][i] = new_class_id
        self._class_to_sample[idx].append([i])

        return new_class_id
