import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

from typing import Dict

from .estimator import Estimator


class LexicalSimilarity(Estimator):
    def __init__(self, metric: str = 'rougeL'):
        self.metric = metric
        super().__init__(['blackbox_sample_texts'], 'sequence')

    def __str__(self):
        return f'LexicalSimilarity_{self.metric}'

    def _score_single(self, t1: str, t2: str):
        if self.metric.startswith('rouge'):
            return rouge_scorer.RougeScorer([self.metric], use_stemmer=True).score(t1, t2)[self.metric].fmeasure
        elif self.metric == 'BLEU':
            return -(1 - sentence_bleu([t1.split()], t2.split())) ** 2
        else:
            raise Exception(f'Unknown metrics for lexical similarity: {self.metric}')

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_texts = stats['blackbox_sample_texts']
        res = []
        for texts in batch_texts:
            sims = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sims.append(self._score_single(texts[i], texts[j]))
            res.append(-np.mean(sims))
        return np.array(res)
