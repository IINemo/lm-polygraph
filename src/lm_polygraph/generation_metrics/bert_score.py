import numpy as np
from sentence_transformers import SentenceTransformer, util

from typing import List, Dict
from .generation_metric import GenerationMetric


class BertScoreMetric(GenerationMetric):
    def __init__(self, lang='en'):
        super().__init__(['greedy_texts'], 'sequence')
        # self.sbert_name = sbert_name
        # self.sbert = SentenceTransformer("all-mpnet-base-v2")
        self.lang = lang

    def __str__(self):
        return f'Bert'

    def _score_single(self, t1: str, t2: str):
        return util.cos_sim(self.sbert.encode(t1), self.sbert.encode(t2)).item()

    def __call__(self, stats: Dict[str, np.ndarray], target_texts: List[str],
                 target_tokens: List[List[int]]) -> np.ndarray:
        return np.array([self._score_single(hyp, ref) for hyp, ref in zip(stats['greedy_texts'], target_texts)])


if __name__ == '__main__':
    from evaluate import load

    bertscore = load("bertscore")
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    results = bertscore.compute(predictions=predictions, references=references, lang="en")

    # """
    # Kind of tests, while there is no test suite
    # """
    # metric = SbertMetric('sbert')
    # stats = {
    #     'greedy_texts': [
    #         "Apple",
    #         "Orange",
    #         "Car",
    #         "The best drink is a beer",
    #         "January is before February"
    #     ]
    # }
    # target_texts = [
    #     "Apple", "Apple", "Apple",
    #     "Octoberfest", "Octoberfest"
    # ]
    #
    # scores = metric(stats, target_texts, None)
    # print(scores)
    #
    # assert scores.shape == (5,)
    # assert scores[0] - 1 < 1e-5
    # assert scores[1] > scores[2]
    # assert scores[3] > scores[4]
