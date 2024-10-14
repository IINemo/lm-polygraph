import numpy as np

from typing import Dict

from .estimator import Estimator


class SentenceSAR(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Sentence SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the sum of the probability of the generated text and text relevance relative to all other generations.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "SentenceSAR"

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

        sentenceSAR = []
        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = np.exp(np.array(sample_log_probs))
            R_s = (
                sample_probs
                * sample_sentence_similarity
                * (1 - np.eye(sample_sentence_similarity.shape[0]))
            )
            sent_relevance = R_s.sum(-1) / self.t
            E_s = -np.log(sent_relevance + sample_probs)
            sentenceSAR.append(E_s.mean())

        return np.array(sentenceSAR)


class OtherSentenceSAR(Estimator):
    """
    Like SAR, but only looks at other samples for each sample in the output.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "OtherSentenceSAR"

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

        sentenceSAR = []
        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = np.exp(np.array(sample_log_probs))
            R_s = (
                sample_probs
                * sample_sentence_similarity
                * (1 - np.eye(sample_sentence_similarity.shape[0]))
            )
            sent_relevance = R_s.sum(-1) / self.t
            E_s = -np.log(sent_relevance)
            sentenceSAR.append(E_s.mean())

        return np.array(sentenceSAR)


class ReweightedSentenceSAR(Estimator):
    """
    Like SAR, but normalizes similarity-based scores at each iteration
    alpha_ij = g(s_i, s_j) / (\sum_k^(K - 1) g(s_i, s_k))
    K - number of samples in output minus one
    """
    def __init__(self, verbose: bool = False):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "ReweightedSentenceSAR"
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        sentenceSAR = []

        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            # Compute probabilities from log probabilities
            sample_probs = np.exp(np.array(sample_log_probs))
            
            # Initialize alpha_ij (reweighted sentence similarities)
            alpha_ij = np.zeros_like(sample_sentence_similarity)

            # Normalize similarity-based scores at each iteration 
            for i in range(sample_sentence_similarity.shape[0]):
                similarity_row = sample_sentence_similarity[i]
                # Exclude self-similarity g(s_i, s_i)
                similarity_row_without_self = similarity_row * (1 - np.eye(len(similarity_row)))[i]
                sum_similarity = np.sum(similarity_row_without_self)
                
                if sum_similarity > 0:
                    alpha_ij[i] = similarity_row_without_self / sum_similarity
                else:
                    alpha_ij[i] = similarity_row_without_self  # If the normalization factor is 0, leave the row unchanged

            # Compute sentence relevance using normalized alpha_ij
            R_s = sample_probs * alpha_ij
            sent_relevance = R_s.sum(-1) / self.t

            # Compute SentenceSAR (Uncertainty Estimation)
            E_s = -np.log(sent_relevance + sample_probs)
            sentenceSAR.append(E_s.mean())

        return np.array(sentenceSAR)



class PPLSentenceSAR(Estimator):
    """
    Like SAR, but uses log probs normalized by sample length in tokens to calculate PPL (Perplexity).
    Tokenwise log-likelihoods are available in stats['sample_log_likelihoods'].
    """
    def __init__(self, verbose: bool = False):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "PPLSentenceSAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the PPL-based sentence-level uncertainty using token-wise log-likelihoods.

        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics, including:
                * 'sample_log_likelihoods': token-wise log-likelihoods for each sample.
        
        Returns:
            np.ndarray: float PPL values for each sample.
                Lower values indicate less uncertainty (better predictions), higher values indicate more uncertainty.
        """
        # Extract token-wise log-likelihoods from the stats
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        sentenceSAR = []

        # Loop over each sample's log-likelihoods and sentence similarities
        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            # Calculate the number of tokens (length of the sample in tokens)

            token_log_likelihoods = np.exp([np.mean(token_ll) for token_ll in sample_log_likelihoods])

            # Initialize the sentence relevance (R_s) using PPL
            R_s = (
                ppl  # Use PPL instead of probabilities
                * sample_sentence_similarity
                * (1 - np.eye(sample_sentence_similarity.shape[0]))  # Remove self-similarity
            )

            # Compute sentence relevance
            sent_relevance = R_s.sum(-1) / self.t
            sample_probs = np.exp(np.array(sample_log_likelihoods))
            # Compute SentenceSAR (Uncertainty Estimation) using PPL
            E_s = -np.log(sent_relevance + ppl)
            sentenceSAR.append(E_s.mean())

        return np.array(sentenceSAR)
