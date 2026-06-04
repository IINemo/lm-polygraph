import numpy as np
import torch

from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    """
    Wrapper around SentenceTransformer that manages model loading and encoding
    arguments. Analogous to the Deberta wrapper — intended to be constructed
    once and shared across stat calculators via BuilderEnvironmentStatCalculator.
    """

    _DEFAULT_ENCODING_ARGUMENTS: Dict = {
        "batch_size": 256,
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "show_progress_bar": False,
    }

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        encoding_arguments: Optional[Dict] = None,
    ):
        self.model_name = model_name
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.cache_folder = cache_folder
        self.encoding_arguments = {
            **self._DEFAULT_ENCODING_ARGUMENTS,
            **(encoding_arguments or {}),
        }
        self._model = None
        self.setup()

    def setup(self):
        if self._model is not None:
            return

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_folder,
        )

    def encode(self, samples: List[str]) -> np.ndarray:
        return self._model.encode(samples, **self.encoding_arguments)
