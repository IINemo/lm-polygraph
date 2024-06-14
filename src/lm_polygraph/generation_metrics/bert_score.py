import numpy as np
from bert_score import BERTScorer

from typing import List, Dict
from .generation_metric import GenerationMetric


class BertScoreMetric(GenerationMetric):
    """
    Calculates BERTScore metric (https://arxiv.org/abs/1904.09675)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, lang="en"):
        super().__init__(["greedy_texts"], "sequence")
        self.scorer = BERTScorer(lang=lang)

    def __str__(self):
        return "Bert"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates BERTScore(https://arxiv.org/abs/1904.09675) between
        stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of BERT Scores for each sample in input.
        """
        scores = self.scorer.score(stats["greedy_texts"], target_texts)[0].numpy()
        return scores


if __name__ == "__main__":
    """
    Kind of tests, while there is no test suite
    """
    metric = BertScoreMetric()
    stats = {
        "greedy_texts": [
            "Apple",
            "Orange",
            "Car",
            "Beer fun in Germany",
            "January is before February",
        ]
    }
    target_texts = ["Apple", "Apple", "Apple", "Octoberfest", "Octoberfest"]

    scores = metric(stats, target_texts, None)
    print(scores)

    assert scores.shape == (5,)
    assert scores[0] - 1 < 1e-5
    assert scores[1] > scores[2]
    assert scores[3] > scores[4]
