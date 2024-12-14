from typing import Dict, List, Tuple
import numpy as np

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model


class InitialStateCalculator(StatCalculator):
    """
    Preprocesses input texts by passing them through unchanged.
    This calculator ensures that the 'input_texts' dependency is always satisfied.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.

        Returns:
            Tuple containing:
                - List of statistics provided by this calculator
                - List of dependencies required by this calculator
        """
        return ["input_texts"], []

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Processes input texts and returns them unchanged.

        Parameters:
            dependencies (Dict[str, np.ndarray]): Input statistics (not used in this calculator).
            texts (List[str]): Input texts batch.
            model: Model (not used in this calculator).
            max_new_tokens (int): Maximum number of new tokens (not used in this calculator).

        Returns:
            Dict[str, List[str]]: Dictionary with the 'input_texts' key and the unchanged input texts as value.
        """
        return {"input_texts": texts}
