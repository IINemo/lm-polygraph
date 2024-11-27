import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.deberta import Deberta
from collections import defaultdict
import torch.nn as nn
import string


def _eval_nli_model(nli_queue: List[Tuple[str, str]], deberta: Deberta) -> List[str]:
    nli_set = list(set(nli_queue))

    softmax = nn.Softmax(dim=1)
    w_probs = defaultdict(lambda: defaultdict(lambda: None))
    for k in range(0, len(nli_set), deberta.batch_size):
        batch = nli_set[k : k + deberta.batch_size]
        encoded = deberta.deberta_tokenizer.batch_encode_plus(
            batch, padding=True, return_tensors="pt"
        ).to(deberta.device)
        logits = deberta.deberta(**encoded).logits
        logits = logits.detach().to(deberta.device)
        for (wi, wj), prob in zip(batch, softmax(logits).cpu().detach()):
            w_probs[wi][wj] = prob

    classes = []
    for w1, w2 in nli_queue:
        pr = w_probs[w1][w2]
        id = pr.argmax()
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        if id == ent_id:
            str_class = "entail"
        elif id == contra_id:
            str_class = "contra"
        else:
            str_class = "neutral"
        classes.append(str_class)
    return classes


class SampleAlternativesNLICalculator(StatCalculator):
    def __init__(self, nli_model):
        super().__init__(
            [
                "sample_tokens_alternatives_nli",
            ],
            ["sample_tokens_alternatives"],
        )

        self.nli_model = nli_model

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch_alternatives = dependencies["sample_tokens_alternatives"]
        batch_alternatives_nli = []
        for samples_alternatives in batch_alternatives:
            sample_alternatives_nli = []
            for sample_alternatives in samples_alternatives:
                nli_matrixes = []
                for w_number, word_alternatives in enumerate(sample_alternatives):
                    nli_queue = []
                    nli_matrix = [
                        ["" for _ in range(len(word_alternatives))]
                        for _ in range(len(word_alternatives))
                    ]
                    if len(word_alternatives) > 0 and not isinstance(
                        word_alternatives[0][0],
                        str,
                    ):
                        word_alternatives = [
                            (model.tokenizer.decode([alt]), prob)
                            for alt, prob in word_alternatives
                        ]
                    words = [self._strip(alt[0]) for alt in word_alternatives]
                    for wi in words:
                        nli_queue.append((words[0], wi))
                        nli_queue.append((wi, words[0]))

                    nli_classes = _eval_nli_model(nli_queue, self.nli_model)
                    nli_class = defaultdict(lambda: None)
                    for nli_cl, (w1, w2) in zip(nli_classes, nli_queue):
                        nli_class[w1, w2] = nli_cl

                    for i, wi in enumerate(words):
                        for j, wj in enumerate(words):
                            # Only calculate NLI with sample token
                            if i > 0 and j > 0:
                                continue
                            nli_matrix[i][j] = nli_class[wi, wj]

                    nli_matrixes.append(nli_matrix)
                sample_alternatives_nli.append(nli_matrixes)
            batch_alternatives_nli.append(sample_alternatives_nli)

        return {"sample_tokens_alternatives_nli": batch_alternatives_nli}
