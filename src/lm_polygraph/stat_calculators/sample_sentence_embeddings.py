import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.utils.sentence_embedder import SentenceEmbedder


class SampleSentenceEmbeddingsCalculator(StatCalculator):
    """
    Encodes each sampled generation with an external sentence embedding model.
    Produces `sample_sentence_embeddings`: List[np.ndarray] of shape (n_samples, embedding_dim).
    Used by PredictiveKernelEntropy and any other estimator that needs dense
    embeddings of sampled texts rather than the LLM's own hidden states.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["sample_sentence_embeddings"], ["sample_texts"]

    def __init__(self, model: SentenceEmbedder):
        super().__init__()
        self._embedder = model

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        return {
            "sample_sentence_embeddings": [
                self._embedder.encode(samples)
                for samples in dependencies["sample_texts"]
            ]
        }
