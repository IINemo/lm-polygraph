import sys

import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

from .estimator import Estimator
from transformers import DebertaForSequenceClassification, DebertaTokenizer


class SemanticEntropy(Estimator):
    def __init__(
            self,
            deberta_path: str = 'microsoft/deberta-large-mnli',
            batch_size: int = 10,
            verbose: bool = False
    ):
        super().__init__(['sample_log_probs', 'sample_texts'], 'sequence')
        self.batch_size = batch_size
        self.init_deberta(deberta_path)
        self.verbose = verbose

    def init_deberta(self, deberta_path):
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            deberta_path, problem_type="multi_label_classification")
        self.deberta_tokenizer = DebertaTokenizer.from_pretrained(deberta_path)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.deberta.to(self.device)
        self.deberta.eval()

        self._sample_to_class = {}
        self._class_to_sample: Dict[int, List] = defaultdict(list)
        self._is_entailment = {}

    def __str__(self):
        return 'SemanticEntropy'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        loglikelihoods_list = stats['sample_log_probs']
        hyps_list = stats['sample_texts']
        return self.batched_call(hyps_list, loglikelihoods_list)

    def batched_call(self, hyps_list: List[List[str]], loglikelihoods_list: List[List[float]],
                     log_weights: Optional[List[List[float]]] = None) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
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
        self._is_entailment = {}

        generators = [self._determine_class(idx, i, hyp)
                      for idx, hyp in enumerate(hyps_list)
                      for i in range(len(hyp))]
        rng = zip(*generators)
        if self.verbose:
            max_len = max(len(hyp) for hyp in hyps_list)
            rng = tqdm(rng, total=max_len, desc='DeBERTa inference')
        for queries in rng:
            not_nan_queries = [(t[0], t[1]) for t in queries if t is not None]
            if len(not_nan_queries) == 0:
                break
            ent = self.is_entailment(not_nan_queries)
            for t in queries:
                if t is None:
                    continue
                self._is_entailment[t[0], t[1]] = ent[0]
                ent = ent[1:]

        return self._sample_to_class, self._class_to_sample

    def is_entailment(self, texts: List[Tuple[str, str]]):
        res = []
        for i in range(0, len(texts), self.batch_size):
            texts1 = [t1 for t1, _ in texts[i:i + self.batch_size]]
            texts2 = [t2 for _, t2 in texts[i:i + self.batch_size]]
            encoded = self.deberta_tokenizer.batch_encode_plus(
                ['[CLS] {} [SEP] {} [SEP]'.format(t1, t2) for t1, t2 in zip(texts1, texts2)],
                add_special_tokens=True, padding='max_length',
                truncation=True, return_attention_mask=True, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                if self.verbose:
                    sys.stderr.write('Inference...')
                    sys.stderr.flush()
                logits = self.deberta(**inp).logits
                if self.verbose:
                    sys.stderr.write('Done')
                    sys.stderr.flush()
            res.append((logits.argmax(-1) == self.deberta.config.label2id['ENTAILMENT']).cpu().numpy())
        return np.concatenate(res)

    def _determine_class(self, idx: int, i: int, texts: List[str]):
        if i == 0:
            self._class_to_sample[idx] = [[0]]
            self._sample_to_class[idx] = {0: 0}
            while True:
                yield None

        cur_text = texts[i]
        found_class = False
        class_id = 0
        while True:
            if class_id >= len(self._class_to_sample[idx]):
                if i - 1 not in self._sample_to_class[idx].keys():
                    yield None
                    continue
                else:
                    break
            class_text = texts[self._class_to_sample[idx][class_id][0]]
            if (class_text, cur_text) not in self._is_entailment.keys():
                yield class_text, cur_text
            if (cur_text, class_text) not in self._is_entailment.keys():
                yield cur_text, class_text
            if self._is_entailment[class_text, cur_text] and self._is_entailment[cur_text, class_text]:
                self._class_to_sample[idx][class_id].append(i)
                self._sample_to_class[idx][i] = class_id
                found_class = True
                break
            class_id += 1

        if not found_class:
            self._sample_to_class[idx][i] = len(self._class_to_sample[idx])
            self._class_to_sample[idx].append([i])
        while True:
            yield None
