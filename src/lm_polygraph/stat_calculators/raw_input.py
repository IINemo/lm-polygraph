import numpy as np
from typing import Dict, List
from .stat_calculator import StatCalculator


class RawInputCalculator(StatCalculator):
    """
    Copies 'input_texts' stat to new 'no_fewshot_input_texts' stat.
    """

    @staticmethod
    def meta_info():
        # outputs, dependencies
        return ["no_fewshot_input_texts"], ["input_texts"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        input_texts = dependencies["input_texts"]
        # Ensure it is copied as numpy array or list as in input
        return {"no_fewshot_input_texts": input_texts}
