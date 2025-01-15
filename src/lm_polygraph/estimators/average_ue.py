import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator

class AveMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "AveMaxprob"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        ave = []
        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = -np.array(sample_log_probs)

            ave.append(sample_probs.mean())

        return np.array(ave)

class AvePPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "AvePPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        ave = []
        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            ppl = -np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])

            ave.append(ppl.mean())

        return np.array(ave)

class AveTokenSAR(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(
            [
                "sample_sentence_similarity",
                "sample_log_likelihoods",
                "sample_token_similarity",
            ],
            "sequence",
        )
        self.verbose = verbose

    def __str__(self):
        return "AveTokenSAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        ave = []
        for i, batch_data in enumerate(zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
        )):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]

            tokenSAR = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                R_t = 1 - token_similarity
                if R_t.sum() == 0:
                    R_t_norm = np.zeros_like(R_t)
                else:
                    R_t_norm = R_t / R_t.sum()
                E_t = -log_likelihoods * R_t_norm
                tokenSAR.append(E_t.sum())
            ave.append(np.mean(tokenSAR))

        return np.array(ave)

class AveMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "sample_entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "AveMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        ave = []
        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            ave.append(np.mean(sample_entropy))

        return np.array(ave)
