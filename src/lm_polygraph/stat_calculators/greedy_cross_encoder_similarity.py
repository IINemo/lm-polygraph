import numpy as np

import itertools
from typing import Dict, List, Tuple
from tqdm import tqdm

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class GreedyCrossEncoderSimilarityMatrixCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity between greedy sequence and sampled sequences.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return (
            [
                "greedy_sentence_similarity_forward",
                "greedy_sentence_similarity_backward",
                "greedy_sentence_similarity",
            ],
            ["input_texts", "sample_texts", "greedy_texts"],
        )

    def __init__(
        self,
        batch_size: int = 10,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-large",
    ):
        super().__init__()
        self.crossencoder_setup = False
        self.batch_size = batch_size
        self.cross_encoder_name = cross_encoder_name

    def _setup(self, device="cuda"):
        self.crossencoder = CrossEncoder(self.cross_encoder_name, device=device)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        device = model.device()

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        batch_texts = dependencies["sample_texts"]
        batch_greedy_texts = dependencies["greedy_texts"]

        batch_pairs = []
        batch_invs = []
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
            sim_scores_f = self.crossencoder.predict(pairs, batch_size=self.batch_size)
            sim_scores_b = self.crossencoder.predict(
                pairs_b, batch_size=self.batch_size
            )

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
