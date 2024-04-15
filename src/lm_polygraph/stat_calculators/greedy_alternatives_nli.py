import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from collections import defaultdict
import torch.nn as nn
import string


class GreedyAlternativesNLICalculator(StatCalculator):
    def __init__(self, nli_model):
        super().__init__(
            [
                "greedy_tokens_alternatives_nli",
            ],
            ["greedy_tokens_alternatives"],
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
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        greedy_alternatives_nli = []
        deberta = self.nli_model
        for sample_alternatives in greedy_alternatives:
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
                    for wj in words:
                        nli_queue.append((wi, wj))
                        nli_queue.append((wj, wi))
                nli_queue = list(set(nli_queue))

                softmax = nn.Softmax(dim=1)
                w_probs = defaultdict(lambda: defaultdict(lambda: None))
                for k in range(0, len(nli_queue), deberta.batch_size):
                    batch = nli_queue[k : k + deberta.batch_size]
                    encoded = deberta.deberta_tokenizer.batch_encode_plus(
                        batch, padding=True, return_tensors="pt"
                    ).to(deberta.device)
                    logits = deberta.deberta(**encoded).logits
                    logits = logits.detach().to(deberta.device)
                    for (wi, wj), prob in zip(batch, softmax(logits).cpu().detach()):
                        w_probs[wi][wj] = prob

                for i, wi in enumerate(words):
                    for j, wj in enumerate(words):
                        pr = w_probs[wi][wj]
                        id = pr.argmax()
                        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
                        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
                        if id == ent_id:
                            str_class = "entail"
                        elif id == contra_id:
                            str_class = "contra"
                        else:
                            str_class = "neutral"
                        nli_matrix[i][j] = str_class

                nli_matrixes.append(nli_matrix)
            greedy_alternatives_nli.append(nli_matrixes)

        return {"greedy_tokens_alternatives_nli": greedy_alternatives_nli}
