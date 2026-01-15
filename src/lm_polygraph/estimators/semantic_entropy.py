import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator


class SemanticEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.model_adapters.whitebox_model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(
        self, verbose: bool = False, class_probability_estimation: str = "sum", samples: str = "all", normalize: bool = False, estimator: str = "sample"
    ):
        self.class_probability_estimation = class_probability_estimation
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
        self.samples = samples
        self.normalize = normalize
        if estimator not in ["sample", "direct"]:
            raise ValueError(
                f"Unknown estimator: {estimator}. Use 'sample' or 'direct'."
            )
        self.estimator = estimator

    def __str__(self):
        base = "SemanticEntropy"
        if self.samples == "unique":
            base += "Unique"
        if self.estimator == "direct":
            base += "Direct"
        if self.normalize:
            base += "Normalized"
        if self.class_probability_estimation == "frequency":
            base += "Empirical"
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
        if self.class_probability_estimation == "sum":
            hyps_list = stats["sample_texts"]
            if self.normalize:
                loglikelihoods_list = [[np.mean(ll) for ll in batch_ll] for batch_ll in stats["sample_log_likelihoods"]]
                # Normalized likelihoods of sequences do not sum to 1, and may give class probabilities > 1 resulting in negative entropy.
                # We don't know the partition function here, so we approximate it by normalizing over the unique samples in the batch. This estimator is consistent but biased.
                indices = [np.unique(hyps, return_index=True)[1] for hyps in hyps_list]
                loglikelihoods_list = [
                    ll - np.logaddexp.reduce(np.array(ll)[ind])
                    for ll, ind in zip(loglikelihoods_list, indices)
                ]
            else:
                loglikelihoods_list = stats["sample_log_probs"]

        elif self.class_probability_estimation == "frequency":
            hyps_list = stats["sample_texts"]
            loglikelihoods_list = None

        index = []
        for i, hyps in enumerate(hyps_list):
            if self.samples == "unique":
                _, ind = np.unique(hyps, return_index=True)
                index.append(ind)
            else:
                index.append(list(range(0, len(hyps))))

            #unique_hyps_list = []
            #unique_ll_list = []

            #for i, hyps in enumerate(hyps_list):
                #hyps, index = np.unique(hyps, return_index=True)
                #unique_hyps_list.append(hyps)
                #if loglikelihoods_list is not None:
                #    ll = np.array(loglikelihoods_list[i])[index]
                #    unique_ll_list.append(ll)

            #hyps_list = unique_hyps_list
            #if loglikelihoods_list is not None:
            #    loglikelihoods_list = unique_ll_list

        self._class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        self._sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        return self.batched_call(index, hyps_list, loglikelihoods_list)

    def batched_call(
        self,
        index: List[List[int]],
        hyps_list: List[List[str]],
        loglikelihoods_list: Optional[List[List[float]]],
    ) -> np.array:
        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            ind = index[i]

            hyps = np.array(hyps_list[i])

            class_to_sample = [list(set(ind).intersection(set(cts))) for cts in self._class_to_sample[i]]
            sample_to_class = {_id: self._sample_to_class[i][_id] for _id in ind}

            if self.class_probability_estimation == "sum":
                ll = np.array(loglikelihoods_list[i])
                class_likelihoods = [
                    np.array(ll[np.array(class_idx)])
                    for class_idx in class_to_sample
                ]
                class_lp = [
                    np.logaddexp.reduce(likelihoods)
                    for likelihoods in class_likelihoods
                ]
            elif self.class_probability_estimation == "frequency":
                num_samples = len(ind)
                class_lp = np.log(
                    [
                        len(class_idx) / num_samples
                        for class_idx in class_to_sample
                    ]
                )

            if self.estimator == "direct":
                semantic_logits[i] = -np.sum(
                    [
                        np.exp(class_lp[sample_to_class[j]]) * class_lp[sample_to_class[j]]
                        for j in ind
                    ]
                )
            else:
                semantic_logits[i] = -np.mean(
                    [
                        class_lp[sample_to_class[j]]
                        for j in ind
                    ]
                )

        entropies = np.array([semantic_logits[i] for i in range(len(hyps_list))])
        clipped_entropies = []

        for ent in entropies:
            if abs(ent) < 1e-8:
                clipped_entropies.append(0.0)
            else:
                clipped_entropies.append(ent)

        return np.array(clipped_entropies)
