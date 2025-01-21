import numpy as np

import itertools
from typing import Dict, List
from tqdm import tqdm

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class GreedySimilarityCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity between greedy sequence and sampled sequences.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "greedy_sentence_similarity_forward",
                "greedy_sentence_similarity_backward",
                "greedy_sentence_similarity",
            ],
            ["input_texts", "sample_texts", "greedy_texts"],
        )

        self.crossencoder_setup = False
        self.nli_model = nli_model

    def _setup(self, device="cuda"):
        self.crossencoder = CrossEncoder(
            "cross-encoder/stsb-roberta-large", device=device
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        device = model.device()
        tokenizer = model.tokenizer

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        batch_texts = dependencies["sample_texts"]
        deberta_batch_size = (
            self.nli_model.batch_size
        )
        batch_input_texts = dependencies["input_texts"]
        batch_greedy_texts = dependencies["greedy_texts"]


        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for texts, greedy_text in zip(batch_texts, batch_greedy_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_invs.append(inv)

        sim_arrays_f = []
        sim_arrays_b = []
        sim_arrays = []
        for i, pairs in tqdm(enumerate(batch_pairs)):
            pairs_b = [(b, a) for a, b in pairs]
            sim_scores_f = self.crossencoder.predict(pairs, batch_size=deberta_batch_size)
            sim_scores_b = self.crossencoder.predict(pairs_b, batch_size=deberta_batch_size)

            inv = batch_invs[i]

            sim_arrays_f.append(sim_scores_f[inv])
            sim_arrays_b.append(sim_scores_b[inv])
            sim_arrays.append((sim_scores_f[inv] + sim_scores_b[inv]) / 2)

        sim_arrays_f = np.stack(sim_arrays_f)
        sim_arrays_b = np.stack(sim_arrays_b)
        sim_arrays = np.stack(sim_arrays)

        return {
            "greedy_sentence_similarity_forward": sim_arrays_f,
            "greedy_sentence_similarity_backward": sim_arrays_b,
            "greedy_sentence_similarity": sim_arrays,
        }
