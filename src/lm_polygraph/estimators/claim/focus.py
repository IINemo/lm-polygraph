import numpy as np
import logging
from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.focus import load_idf, token_level_focus_scores

log = logging.getLogger(__name__)


class FocusClaim(Estimator):
    """
    FocusClaim is a claim-level uncertainty estimator that builds on the Focus method
    by computing hallucination scores for individual claims within generated text.

    This variant computes per-claim uncertainty scores using attention patterns,
    token probabilities, IDF weighting, and linguistic importance (e.g. NER/POS),
    as proposed in:
    "Hallucination Detection in Neural Text Generation via Focused Uncertainty Estimation"
    (https://arxiv.org/abs/2311.13230).

    Args:
        gamma (float): Context penalty coefficient controlling influence of preceding tokens.
        p (float): Probability threshold; token predictions below this are masked.
        model_name (str): Hugging Face tokenizer model name or path.
        path (str): Path to save/load precomputed IDF values.
        idf_dataset (str): Dataset name for IDF computation.
        trust_remote_code (bool): Whether to trust custom dataset loading code.
        idf_seed (int): Seed for dataset shuffling/sampling.
        idf_dataset_size (int): Number of samples used for IDF (-1 for all).
        spacy_path (str): spaCy language model name or path for NER/POS tagging.
    """

    def __init__(
        self,
        gamma: float,
        p: float,
        model_name: str,
        path: str,
        idf_dataset: str,
        trust_remote_code: bool,
        idf_seed: int,
        idf_dataset_size: int,
        spacy_path: str,
    ):
        super().__init__(
            [
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_texts",
                "claims",
                "attention_all",
                "tokenizer",
            ],
            "claim",
        )
        self.p = p
        self.gamma = gamma
        self.idf_stats = load_idf(
            model_name,
            path,
            idf_dataset,
            trust_remote_code,
            idf_seed,
            idf_dataset_size,
            spacy_path,
        )

    def __str__(self):
        return f"FocusClaim (gamma={self.gamma})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> List[List[float]]:
        """
        Compute claim-level Focus uncertainty scores using token-level scores
        and aligned token indices per claim.

        Args:
            stats (Dict[str, np.ndarray]): Dictionary containing generation outputs,
                including token probabilities, attention maps, and claim-token alignments.

        Returns:
            List[List[float]]: Nested list of uncertainty scores, where each inner list
                corresponds to the scores of claims in a single sequence.
        """
        claims = stats["claims"]

        all_token_focus, all_kw_mask = token_level_focus_scores(
            stats,
            self.idf_stats,
            self.p,
            self.gamma,
        )

        focus_ue = []
        for token_focus, claims_i in zip(all_token_focus, claims):
            focus_ue.append([])
            for claim in claims_i:
                tokens = np.array(claim.aligned_token_ids).astype(int)
                claim_p_i = np.array(token_focus)[tokens]
                focus_ue[-1].append(claim_p_i.mean())
        return focus_ue
