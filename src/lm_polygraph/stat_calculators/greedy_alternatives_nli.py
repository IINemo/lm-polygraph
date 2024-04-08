import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from collections import defaultdict
import torch.nn as nn
import string


class GreedyTokensAlternativesNLICalculator(StatCalculator):
    def __init__(self, llm_tokenizer, nli_model, nli_batch_size: int = 100):
        super().__init__(
            [
                "greedy_tokens_alternatives_nli",
            ],
            [
                "greedy_tokens_alternatives",
            ],
        )
        self.nli_batch_size = nli_batch_size
        self.nli_model = nli_model
        self.llm_tokenizer = llm_tokenizer

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str] = None,
        model: WhiteboxModel = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        greedy_alternatives_nli = []
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
                        (self.llm_tokenizer.decode([alt]), prob)
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
                for k in range(0, len(nli_queue), self.nli_batch_size):
                    batch = nli_queue[k : k + self.nli_batch_size]
                    encoded = self.nli_model.tokenizer.batch_encode_plus(
                        batch, padding=True, return_tensors="pt"
                    ).to(self.nli_model.device)
                    logits = self.nli_model(**encoded).logits
                    logits = logits.detach().to(self.nli_model.device)
                    for (wi, wj), prob in zip(batch, softmax(logits).cpu().detach()):
                        w_probs[wi][wj] = prob

                for i, wi in enumerate(words):
                    for j, wj in enumerate(words):
                        pr = w_probs[wi][wj]
                        id = pr.argmax()
                        ent_id = self.nli_model.config.label2id["ENTAILMENT"]
                        contra_id = self.nli_model.config.label2id["CONTRADICTION"]
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
