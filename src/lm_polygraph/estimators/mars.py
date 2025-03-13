from transformers import AutoModel, AutoTokenizer
import numpy as np

from typing import Dict

from .estimator import Estimator

class MARS(Estimator):
    def __init__(self, verbose=False):
        super().__init__(
            [
                "sample_sentence_similarity",
                "sample_log_likelihoods",
                "sample_token_similarity",
            ],
            "sequence",
        )
        self.verbose = verbose
        self.t = 0.001

    def calculate_mars_token_score(self, log_likelihoods, token_similarity):
        
        mars_token_score = log_likelihoods * token_similarity
        return mars_token_score

    def calculate_mars_sentence_relevance(self, mars_token_scores, sentence_similarity):
        mars_sentence_relevance = mars_token_scores.mean() * sentence_similarity.mean()
        return mars_sentence_relevance

    def calculate_mars_sample_score(self, mars_sentence_relevance):
        mars_sample_score = mars_sentence_relevance
        return mars_sample_score

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        MARS_scores = []
        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]

            mars_token_scores = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                mars_token_score = self.calculate_mars_token_score(log_likelihoods, token_similarity)  
                mars_token_scores.append(mars_token_score)

            mars_token_scores = np.array(mars_token_scores)
            mars_sentence_relevance = self.calculate_mars_sentence_relevance(mars_token_scores, sample_sentence_similarity)
            mars_sample_score = self.calculate_mars_sample_score(mars_sentence_relevance)
            MARS_scores.append(mars_sample_score)

        return np.array(MARS_scores)
