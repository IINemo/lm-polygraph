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
                "greedy_sentence_similarity",
            ],
            ["input_texts", "sample_tokens", "sample_texts", "greedy_tokens", "greedy_texts"],
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

        batch_sample_tokens = dependencies["sample_tokens"]
        batch_texts = dependencies["sample_texts"]
        deberta_batch_size = (
            self.nli_model.batch_size
        )
        batch_input_texts = dependencies["input_texts"]
        batch_greedy_tokens = dependencies["greedy_tokens"]
        batch_greedy_texts = dependencies["greedy_texts"]

        special_tokens = list(model.tokenizer.added_tokens_decoder.keys())

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for texts, greedy_text in zip(batch_texts, batch_greedy_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_invs.append(inv)

        sim_arrays = []
        for i, pairs in tqdm(enumerate(batch_pairs)):
            sim_scores = self.crossencoder.predict(pairs, batch_size=deberta_batch_size)

            inv = batch_invs[i]

            sim_arrays.append(sim_scores[inv])

        sim_arrays = np.stack(sim_arrays)

        return {
            "greedy_sentence_similarity": sim_arrays,
        }
