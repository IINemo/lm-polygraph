import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.utils.nli_semantic_matrix import calculate_semantic_matrix


class SemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "semantic_matrix_entail",
            "semantic_matrix_contra",
            "semantic_matrix_classes",
            "semantic_matrix_entail_logits",
            "semantic_matrix_contra_logits",
            "entailment_id",
        ], ["sample_texts"]

    def __init__(self, nli_model):
        super().__init__()
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the NLI semantic matrix for generation samples using DeBERTa model.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'sample_texts' (List[List[str]]): several sampling generations
                    for each input text in the batch.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'semantic_matrix_entail' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with probabilities of 'ENTAILMENT' output of DeBERTa.
                - 'semantic_matrix_contra' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with probabilities of 'CONTRADICTION' output of DeBERTa.
                - 'semantic_matrix_entail_logits' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with logits of 'ENTAILMENT' output of DeBERTa.
                - 'semantic_matrix_contra_logits' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with logits of 'CONTRADICTION' output of DeBERTa.
                - 'semantic_matrix_classes' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with the NLI label id corresponding to the DeBERTa prediction.
        """

        return calculate_semantic_matrix(self.nli_model, dependencies["sample_texts"])
