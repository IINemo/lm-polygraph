import numpy as np
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.utils.nli_semantic_matrix import calculate_semantic_matrix


class SDLGSemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for SDLG generation samples using an NLI model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "sdlg_semantic_matrix_entail",
            "sdlg_semantic_matrix_contra",
            "sdlg_semantic_matrix_classes",
            "sdlg_semantic_matrix_entail_logits",
            "sdlg_semantic_matrix_contra_logits",
            "sdlg_entailment_id",
        ], ["sdlg_sample_texts"]

    def __init__(self, nli_model):
        super().__init__()
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        matrix = calculate_semantic_matrix(
            self.nli_model, dependencies["sdlg_sample_texts"]
        )
        return {f"sdlg_{key}": value for key, value in matrix.items()}
