import numpy as np

import itertools
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class CrossEncoderSimilarityMatrixVisualCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity matrix for generation samples using RoBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "sample_sentence_similarity",
            "sample_token_similarity",
            "token_similarity",
        ], ["input_texts", "sample_tokens", "sample_texts", "greedy_tokens"]

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
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        device = model.device()
        tokenizer = model.processor_visual

        if not self.crossencoder_setup:
            self._setup(device=device)
            self.crossencoder_setup = True

        batch_sample_tokens = dependencies["sample_tokens"]
        batch_texts = dependencies["sample_texts"]
        batch_input_texts = dependencies["input_texts"]
        batch_greedy_tokens = dependencies["greedy_tokens"]

        special_tokens = list(tokenizer.tokenizer.added_tokens_decoder.keys())

        def normalize_text(text):
            if isinstance(text, list):
                return " ".join([str(t) for t in text])
            return str(text)

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
            input_texts = normalize_text(input_texts)

            if len(tokens) > 1:
                is_special_tokens = np.isin(tokens, special_tokens)
                cropped_tokens = list(itertools.combinations(tokens, len(tokens) - 1))[
                    ::-1
                ]
                raw_text = (
                    " ".join(input_texts)
                    + " "
                    + tokenizer.tokenizer.decode(tokens, skip_special_tokens=True)
                )
                batches = [
                    (
                        raw_text,
                        " ".join(input_texts)
                        + " "
                        + tokenizer.tokenizer.decode(t, skip_special_tokens=True),
                    )
                    for t in cropped_tokens
                ]
                token_scores = self.crossencoder.predict(
                    batches, batch_size=self.batch_size
                )
                token_scores[is_special_tokens] = 1
            else:
                token_scores = np.array([0.5] * len(tokens))
            batch_token_scores.append(token_scores)

        sim_matrices = []
        for i, pairs in enumerate(batch_pairs):
            sim_scores = self.crossencoder.predict(pairs, batch_size=self.batch_size)
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
            input_texts = normalize_text(input_texts)

            for tokens in sample_tokens:
                if len(tokens) > 1:
                    is_special_tokens = np.isin(tokens, special_tokens)
                    cropped_tokens = list(
                        itertools.combinations(tokens, len(tokens) - 1)
                    )[::-1]
                    raw_text = (
                        input_texts
                        + " "
                        + tokenizer.tokenizer.decode(tokens, skip_special_tokens=True)
                    )
                    batches = [
                        (
                            raw_text,
                            input_texts
                            + " "
                            + tokenizer.tokenizer.decode(
                                list(t), skip_special_tokens=True
                            ),
                        )
                        for t in cropped_tokens
                    ]
                    token_scores = self.crossencoder.predict(
                        batches, batch_size=self.batch_size
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
