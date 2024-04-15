import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class CrossEncoderSimilarityMatrixCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity matrix for generation samples using RoBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "sample_sentence_similarity",
                "sample_token_similarity",
                "token_similarity",
            ],
            ["input_texts", "sample_tokens", "sample_texts", "greedy_tokens"],
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
        )  # TODO: Why we use parameters of nli_model for the cross-encoder model???
        batch_input_texts = dependencies["input_texts"]
        batch_greedy_tokens = dependencies["greedy_tokens"]

        special_tokens = list(model.tokenizer.added_tokens_decoder.keys())

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for texts in batch_texts:
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product(unique_texts, unique_texts)))
            batch_invs.append(inv)
            batch_counts.append(len(unique_texts))

        batch_token_scores = []
        for input_texts, tokens in zip(batch_input_texts, batch_greedy_tokens):
            if len(tokens) > 1:
                is_special_tokens = np.isin(tokens, special_tokens)
                cropped_tokens = list(itertools.combinations(tokens, len(tokens) - 1))[
                    ::-1
                ]
                raw_text = (
                    input_texts
                    + " "
                    + tokenizer.decode(tokens, skip_special_tokens=True)
                )
                batches = [
                    (
                        raw_text,
                        input_texts
                        + " "
                        + tokenizer.decode(list(t), skip_special_tokens=True),
                    )
                    for t in cropped_tokens
                ]
                token_scores = self.crossencoder.predict(
                    batches, batch_size=deberta_batch_size
                )
                token_scores[is_special_tokens] = 1
            else:
                token_scores = np.array([0.5] * len(tokens))
            batch_token_scores.append(token_scores)

        sim_matrices = []
        for i, pairs in enumerate(batch_pairs):
            sim_scores = self.crossencoder.predict(pairs, batch_size=deberta_batch_size)
            unique_mat_shape = (batch_counts[i], batch_counts[i])

            sim_scores_matrix = sim_scores.reshape(unique_mat_shape)
            inv = batch_invs[i]

            # Recover full matrices from unques by gathering along both axes
            # using inverse index
            sim_matrices.append(sim_scores_matrix[inv, :][:, inv])
        sim_matrices = np.stack(sim_matrices)

        batch_samples_token_scores = []
        for sample_tokens, input_texts in zip(batch_sample_tokens, batch_input_texts):
            samples_token_scores = []
            for tokens in sample_tokens:
                if len(tokens) > 1:
                    is_special_tokens = np.isin(tokens, special_tokens)
                    cropped_tokens = list(
                        itertools.combinations(tokens, len(tokens) - 1)
                    )[::-1]
                    raw_text = (
                        input_texts
                        + " "
                        + tokenizer.decode(tokens, skip_special_tokens=True)
                    )
                    batches = [
                        (
                            raw_text,
                            input_texts
                            + " "
                            + tokenizer.decode(list(t), skip_special_tokens=True),
                        )
                        for t in cropped_tokens
                    ]
                    token_scores = self.crossencoder.predict(
                        batches, batch_size=deberta_batch_size
                    )
                    token_scores[is_special_tokens] = 1
                else:
                    token_scores = np.array([0.5] * len(tokens))
                samples_token_scores.append(token_scores)
            batch_samples_token_scores.append(samples_token_scores)

        return {
            "sample_sentence_similarity": sim_matrices,
            "sample_token_similarity": batch_samples_token_scores,
            "token_similarity": batch_token_scores,
        }
