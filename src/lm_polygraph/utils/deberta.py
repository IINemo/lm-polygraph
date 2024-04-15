import torch
import numpy as np

from transformers import DebertaForSequenceClassification, DebertaTokenizer
from typing import Dict

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class Deberta:
    """
    Allows for the implementation of a singleton DeBERTa model which can be shared across
    different uncertainty estimation methods in the code.
    """

    def __init__(
        self,
        deberta_path: str = "microsoft/deberta-large-mnli",
        batch_size: int = 10,
        device=None,
    ):
        """
        Parameters
        ----------
        deberta_path : str
            huggingface path of the pretrained DeBERTa (default 'microsoft/deberta-large-mnli')
        device : str
            device on which the computations will take place (default 'cuda:0' if available, else 'cpu').
        """
        #super().__init__(["deberta"], [])
        self.deberta_path = deberta_path
        self.batch_size = batch_size
        self.deberta = None
        self.deberta_tokenizer = None
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def to(self, device):
        self.device = device
        if self.deberta is not None:
            self.deberta.to(self.device)

    def setup(self):
        """
        Loads and prepares the DeBERTa model from the specified path.
        """
        if self.deberta is not None:
            return
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            self.deberta_path, problem_type="multi_label_classification"
        )
        self.deberta_tokenizer = DebertaTokenizer.from_pretrained(self.deberta_path)
        self.deberta.to(self.device)
        self.deberta.eval()

    # def __call__(
    #     self,
    #     *args,
    #     **kwargs,
    # ) -> Dict[str, np.ndarray]:
    #     self.setup()
    #     return {"deberta": self}
