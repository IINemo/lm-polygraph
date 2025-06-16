import numpy as np
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.utils.semantic_classes_utils import calculate_semantic_classes


class SDLGSemanticClassesCalculator(StatCalculator):
    """
    Partitions SDLG generation samples into semantic classes using NLI semantic logic.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "sdlg_semantic_classes_entail",
        ], [
            "sdlg_sample_texts",
            "sdlg_semantic_matrix_classes",
            "sdlg_entailment_id",
        ]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        sample_to_class, class_to_sample = calculate_semantic_classes(
            dependencies["sdlg_sample_texts"],
            dependencies["sdlg_semantic_matrix_classes"],
            dependencies["sdlg_entailment_id"],
        )
        return {
            "sdlg_semantic_classes_entail": {
                "sample_to_class": sample_to_class,
                "class_to_sample": class_to_sample,
            }
        }
