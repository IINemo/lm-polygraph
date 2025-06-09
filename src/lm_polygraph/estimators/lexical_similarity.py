"""Lexical similarity-based uncertainty estimation for language models."""

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from typing import Dict, List
from .estimator import Estimator

from absl import logging as absl_logging
import os

# This prevents bullshit spam from rouge scorer
absl_logging.set_verbosity(absl_logging.WARNING)

os.environ["ROUGE_HOME"] = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "scripts", "ROUGE-1.5.5"
)
from rouge import Rouge


class LexicalSimilarity(Estimator):
    """
    Lexical similarity uncertainty estimator for black-box and white-box models.
    
    This estimator measures uncertainty by analyzing the lexical diversity among
    multiple generated samples. Low similarity between different generations for
    the same input indicates high uncertainty, as the model produces varied outputs.
    
    The method works by:
    1. Generating multiple samples for each input (via sampling)
    2. Computing pairwise similarity scores between all samples
    3. Using the average similarity as an inverse uncertainty measure
    
    This approach is particularly valuable because:
    - It works with any model (no need for internal access)
    - Captures both surface-level and semantic uncertainty
    - Can use various similarity metrics (ROUGE-L, ROUGE-1, ROUGE-2)
    - Effective for detecting when models are "making things up"
    
    Attributes:
        dependencies (List[str]): Requires 'sample_texts' statistics
        level (str): Operates at 'sequence' level
        similarity_metric (str): Metric to use ('rougeL', 'rouge1', 'rouge2')
        
    References:
        Fomicheva et al., 2020. "Unsupervised Quality Estimation for Neural Machine
        Translation" (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/)
        
    Examples:
        >>> from lm_polygraph import BlackboxModel
        >>> from lm_polygraph.estimators import LexicalSimilarity
        >>> model = BlackboxModel.from_openai('KEY', 'gpt-3.5-turbo')
        >>> 
        >>> # Using ROUGE-L (recommended)
        >>> estimator = LexicalSimilarity('rougeL')
        >>> result = estimate_uncertainty(
        ...     model, estimator,
        ...     "What happened in 1969?",
        ...     generation_params={"n": 5}  # Generate 5 samples
        ... )
        >>> 
        >>> # Using ROUGE-1 (unigram overlap)
        >>> estimator = LexicalSimilarity('rouge1')
        
    See Also:
        SemanticEntropy: Similarity based on meaning rather than words
        NumSemSets: Counts semantic clusters instead of averaging
        SAR: Self-alignment rate for semantic similarity
        
    Note:
        - Requires multiple samples (num_return_sequences > 1)
        - More samples generally give better estimates
        - ROUGE-L is recommended as it captures longest common subsequences
        - Works best when samples are generated with temperature > 0
    """
    
    def __init__(self, similarity_metric: str = "rougeL"):
        """
        Initialize LexicalSimilarity estimator.
        
        Parameters:
            similarity_metric: Which ROUGE metric to use for similarity:
                - 'rougeL': Longest common subsequence (recommended)
                - 'rouge1': Unigram overlap
                - 'rouge2': Bigram overlap
                Default: 'rougeL'
                
        Raises:
            ValueError: If similarity_metric is not recognized
        """
        if similarity_metric not in ["rougeL", "rouge1", "rouge2"]:
            raise ValueError(
                f"Unknown similarity metric: {similarity_metric}. "
                f"Choose from: 'rougeL', 'rouge1', 'rouge2'"
            )
        self.similarity_metric = similarity_metric
        if self.similarity_metric.startswith("rouge"):
            self.scorer = rouge_scorer.RougeScorer([self.similarity_metric], use_stemmer=True)
        super().__init__(["sample_texts"], "sequence")

    def __str__(self) -> str:
        """Return unique string identifier including the metric used."""
        return f"LexicalSimilarity_{self.similarity_metric}"

    def _score_single(self, t1: str, t2: str):
        if self.similarity_metric.startswith("rouge"):
            return self.scorer.score(t1, t2)[self.similarity_metric].fmeasure
        elif self.similarity_metric == "BLEU":
            min_sentence_len = min(len(t1.split()), len(t2.split()))
            if min_sentence_len == 1:
                weights = [1.0, 0.0, 0.0, 0.0]
            elif min_sentence_len == 2:
                weights = [0.5, 0.5, 0.0, 0.0]
            elif min_sentence_len == 3:
                weights = [0.33, 0.33, 0.33, 0.0]
            else:
                # default weights in sentence_bleu
                weights = [0.25, 0.25, 0.25, 0.25]
            return sentence_bleu([t1.split()], t2.split(), weights=weights)
        else:
            raise Exception(f"Unknown metrics for lexical similarity: {self.similarity_metric}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty based on lexical similarity between samples.
        
        Computes pairwise ROUGE scores between all generated samples and
        returns the negative average similarity as uncertainty. High similarity
        means low uncertainty (consistent outputs), while low similarity means
        high uncertainty (diverse outputs).
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'sample_texts': List of lists, where each inner list contains
                  multiple text samples generated for one input prompt
                   
        Returns:
            np.ndarray: Array of uncertainty scores (negative mean similarity)
                for each input. Shape: (n_sequences,). Higher values indicate
                more diverse outputs and thus higher uncertainty.
                Range typically: [-1, 0] where 0 is maximum uncertainty.
            
        Raises:
            KeyError: If 'sample_texts' is not in stats
            ValueError: If fewer than 2 samples per input
            
        Note:
            The method requires at least 2 samples per input to compute
            meaningful similarities. More samples improve the estimate quality.
        """
        sample_texts = stats["sample_texts"]
        return self.sim_score(sample_texts)

    def sim_score(self, sample_texts: List[List[str]]) -> np.ndarray:
        """
        Compute similarity scores for batches of generated samples.
        
        Parameters:
            sample_texts: List of sample lists for each input
            
        Returns:
            Array of negative mean similarities (uncertainty scores)
        """
        ue = []
        for samples in sample_texts:
            ue.append(np.mean(self.pairwise_similarity(samples)))
        return np.array(ue) * (-1)

    def pairwise_similarity(self, texts: List[str]) -> List[float]:
        """
        Compute pairwise ROUGE similarities between all text pairs.
        
        Parameters:
            texts: List of text samples to compare
            
        Returns:
            List of similarity scores for all unique pairs
        """
        n_samples = len(texts)
        scores = []
        scorer = Rouge()

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if texts[i].strip() == "" or texts[j].strip() == "":
                    scores.append(float(texts[i].strip() == texts[j].strip()))
                else:
                    # Compute ROUGE score for the pair
                    rouge_scores = scorer.get_scores(texts[i], texts[j])
                    f_score = rouge_scores[0][self.similarity_metric]["f"]
                    scores.append(f_score)

        return scores
