import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator


class MaxprobGSU(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        use_log: bool = True,
        reverse: bool = False
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.use_log = use_log
        self.reverse = reverse

    def __str__(self):
        base = "MaxprobGSU"
        if not self.use_log:
            base += "_no_log"
            if self.reverse:
                base += "_reverse"
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the sentenceSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with cross-encoder similarities in 'sample_sentence_similarity'
        Returns:
            np.ndarray: float sentenceSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        GSU = []
        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = np.exp(np.array(sample_log_probs))
            R_s = (
                sample_probs
                * sample_sentence_similarity
            )
            sent_relevance = R_s.sum(-1)

            if self.use_log:
                E_s = -np.log(sent_relevance)
            else:
                E_s = -sent_relevance if self.reverse else sent_relevance

            GSU.append(E_s.mean())

        return np.array(GSU)


class PPLGSU(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        use_log: bool = True,
        reverse: bool = False
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.use_log = use_log
        self.reverse = reverse

    def __str__(self):
        base = "PPLGSU"
        if not self.use_log:
            base += "_no_log"
            if self.reverse:
                base += "_reverse"
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the sentenceSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with cross-encoder similarities in 'sample_sentence_similarity'
        Returns:
            np.ndarray: float sentenceSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        GSU = []
        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            ppl = np.exp([np.mean(token_ll) for token_ll in sample_log_likelihoods])

            R_s = (
                ppl
                * sample_sentence_similarity
            )
            sent_relevance = R_s.sum(-1)

            if self.use_log:
                E_s = -np.log(sent_relevance)
            else:
                E_s = -sent_relevance if self.reverse else sent_relevance

            GSU.append(E_s.mean())

        return np.array(GSU)


class TokenSARGSU(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        use_log: bool = True,
        reverse: bool = False
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
        self.use_log = use_log
        self.reverse = reverse

    def __str__(self):
        base = "TokenSARGSU"
        if not self.use_log:
            base += "_no_log"
            if self.reverse:
                base += "_reverse"
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the SAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) for each sample in 'sample_log_likelihoods'
                * similarity for each sample of the generated text and generated text without one token for each token in 'sample_token_similarity',
                * matrix with cross-encoder similarities in 'sample_sentence_similarity'
        Returns:
            np.ndarray: float SAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        GSU = []
        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]

            tokenSAR = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                R_t = 1 - token_similarity
                R_t_norm = R_t / R_t.sum()
                E_t = -log_likelihoods * R_t_norm
                tokenSAR.append(E_t.sum())

            tokenSAR = np.array(tokenSAR)
            probs_token_sar = np.exp(-tokenSAR)
            R_s = (
                probs_token_sar
                * sample_sentence_similarity
            )
            sent_relevance = R_s.sum(-1)
            E_s = -np.log(sent_relevance)
            if self.use_log:
                E_s = -np.log(sent_relevance)
            else:
                E_s = -sent_relevance if self.reverse else sent_relevance

            GSU.append(E_s.mean())

        return np.array(GSU)


class MTEGSU(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        use_log: bool = True,
        reverse: bool = False
    ):
        super().__init__(["sample_sentence_similarity", "sample_entropy"], "sequence")
        self.verbose = verbose
        self.use_log = use_log
        self.reverse = reverse

    def __str__(self):
        base = "MTEGSU"
        if not self.use_log:
            base += "_no_log"
            if self.reverse:
                base += "_reverse"
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the sentenceSAR for each sample using Mean Token Entropy (MTE).

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * 'sample_entropy': Mean Token Entropy for each sample,
                * 'sample_sentence_similarity': matrix with cross-encoder similarities.
        
        Returns:
            np.ndarray: float sentenceSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        GSU = []
        # Loop over each sample's Mean Token Entropy and sentence similarities
        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            # Use MTE for sentence relevance calculation
            R_s = sample_entropy * sample_sentence_similarity
            
            # Compute sentence relevance by summing along the last axis
            sent_relevance = R_s.sum(-1)

            # Calculate E_s with options for log transformation and reversal
            if self.use_log:
                E_s = -np.log(sent_relevance)
            else:
                E_s = -sent_relevance if self.reverse else sent_relevance

            GSU.append(E_s.mean())

        return np.array(GSU)
