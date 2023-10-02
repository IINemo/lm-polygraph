import sys

import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

from .estimator import Estimator
from .common import DEBERTA


class SemanticEntropy(Estimator):
    def __init__(
            self,
            batch_size: int = 10,
            verbose: bool = False
    ):
        super().__init__(['sample_log_probs', 'sample_texts', 'semantic_matrix_entail'], 'sequence')
        self.batch_size = batch_size
        DEBERTA.setup()
        self.verbose = verbose

    def __str__(self):
        return 'SemanticEntropy'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        loglikelihoods_list = stats['sample_log_probs']

        entailment_id = DEBERTA.deberta.config.label2id['ENTAILMENT']]
        self._is_entailment = (stats['semantic_matrix_classes'] == entailment_id)

        # Concatenate hypos with input texts
        hyps_list = [[] for _ in stats["input_texts"]]
        for i, input_text in enumerate(stats["input_texts"]):
            for hyp in stats["sample_texts"][i]:
                hyps_list[i].append(" ".join([input_text, hyp]))

        return self.batched_call(hyps_list, loglikelihoods_list)

    def batched_call(self, hyps_list: List[List[str]], loglikelihoods_list: List[List[float]],
                     log_weights: Optional[List[List[float]]] = None) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [np.array(loglikelihoods_list[i])[np.array(class_idx)]
                                 for class_idx in self._class_to_sample[i]]
            class_lp = [np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods]
            if log_weights[i] is None:
                log_weights[i] = [0 for _ in hyps_list[i]]
            semantic_logits[i] = -np.mean(
                [class_lp[self._sample_to_class[i][j]] * np.exp(log_weights[i][j])
                 for j in range(len(hyps_list[i]))])
        return np.array([semantic_logits[i] for i in range(len(hyps_list))])

    def get_classes(self, hyps_list: List[List[str]]):
        self._sample_to_class = {}
        self._class_to_sample: Dict[int, List] = defaultdict(list)

        [self._determine_class(idx, i)
         for idx, hyp in enumerate(hyps_list)
         for i in range(len(hyp))]

        return self._sample_to_class, self._class_to_sample

    def _determine_class(self, idx: int, i: int):
        if i == 0:
            self._class_to_sample[idx] = [[0]]
            self._sample_to_class[idx] = {0: 0}

            return 0

        class_id = 0
        for class_id in range(len(self._class_to_sample[idx]):
            class_text_id = self._class_to_sample[idx][class_id][0]
            if self._is_entailment[idx, class_text_id, i] and self._is_entailment[idx, i, class_text_id]:
                self._class_to_sample[idx][class_id].append(i)
                self._sample_to_class[idx][i] = class_id

                return class_id
        
        new_class_id = len(self._class_to_sample[idx])
        self._sample_to_class[idx][i] = new_class_id
        self._class_to_sample[idx].append([i])

        return new_class_id
