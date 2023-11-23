import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

from typing import Dict

from .estimator import Estimator


class LexicalSimilarity(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Lexical Similarity" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    The method calculates mean similarity between all pairs of sampled generations with minus sign.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(self, metric: str = "rougeL"):
        """
        Parameters:
            metric (str): similarity metric (default: 'rougeL'). Possible values:
                * rouge1 / rouge2 / rougeL
                * BLEU
        """
        self.metric = metric
        if self.metric.startswith("rouge"):
            self.scorer = rouge_scorer.RougeScorer([self.metric], use_stemmer=True)
        super().__init__(["blackbox_sample_texts"], "sequence")

    def __str__(self):
        return f"LexicalSimilarity_{self.metric}"

    def _score_single(self, t1: str, t2: str):
        if self.metric.startswith("rouge"):
            return self.scorer.score(t1, t2)[self.metric].fmeasure
        elif self.metric == "BLEU":
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
            raise Exception(f"Unknown metrics for lexical similarity: {self.metric}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the mean similarity with minus sign for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * several sampled texts in 'blackbox_sample_texts'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_texts = stats["blackbox_sample_texts"]
        res = []
        for texts in batch_texts:
            sims = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sims.append(self._score_single(texts[i], texts[j]))
            res.append(-np.mean(sims))
        return np.array(res)
