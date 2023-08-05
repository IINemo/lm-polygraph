import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from .common import DEBERTA
import torch.nn as nn

softmax = nn.Softmax(dim=1)


class NumSemSets(Estimator):
    def __init__(
            self,
            batch_size: int = 10,
            verbose: bool = False,
    ):
        """
        A number of semantic sets in response (higher = bigger uncertainty).

        """
        super().__init__(['blackbox_sample_texts'], 'sequence')
        self.batch_size = batch_size
        DEBERTA.setup()
        self.verbose = verbose

    def __str__(self):
        return f'NumSemSets'

    def get_pairs_semsets(self, lst):
        pairs = []
        for i in range(len(lst) - 1):
            pairs.append((lst[i], lst[i + 1]))
        return pairs

    def U_NumSemSets(self, answers):

        lst = self.get_pairs_semsets(answers)
        # basically we have only 1 semantic set
        num_sets = 1
        device = DEBERTA.deberta.device
        # we iterate over responces and incerase num_sets if the NLI condition is fulfilled
        for (sentence_1, sentence_2) in lst:
            # Tokenize input sentences
            encoded_input_forward = DEBERTA.deberta_tokenizer(sentence_1, sentence_2, return_tensors='pt').to(device)
            encoded_input_backward = DEBERTA.deberta_tokenizer(sentence_2, sentence_1, return_tensors='pt').to(device)

            logits_forward = DEBERTA.deberta(**encoded_input_forward).logits.detach().to(device)
            logits_backward = DEBERTA.deberta(**encoded_input_backward).logits.detach().to(device)

            probs_forward = softmax(logits_forward).to(device)
            probs_backward = softmax(logits_backward).to(device)

            p_entail_forward = probs_forward[0][2]
            p_entail_backward = probs_backward[0][2]

            p_contra_forward = probs_forward[0][0]
            p_contra_backward = probs_backward[0][0]

            if (p_entail_forward > p_contra_forward) & (p_entail_backward > p_contra_backward):
                pass
            else:
                num_sets += 1

        return num_sets

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for answers in stats['blackbox_sample_texts']:
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_NumSemSets(answers))

        return np.array(res)
