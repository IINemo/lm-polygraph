import numpy as np
from typing import Dict, List, Tuple

from ..stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from .utils import flatten, reconstruct


class StepsEntropyCalculator(StatCalculator):
    """
    Calculates entropy of probabilities at each step position in the generation of a Whitebox model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "steps_entropy",
        ], ["sample_steps_log_probs", "claims"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str] = None,
        model: WhiteboxModel = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the entropy of probabilities at each step position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, which includes:
                * 'sample_steps_log_probs' (List[List[List[float]]]): log-probabilities of the generation tokens for each step.
                * 'claims' (List[List[Dict]]): claim information for each sample.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[List[float]]] entropies calculated at 'steps_entropy' key.
        """
        sample_steps_log_probs = dependencies["sample_steps_log_probs"]

        # Flatten the nested structure for processing
        flattened_log_probs = flatten(sample_steps_log_probs)

        steps_entropy = []
        for step_log_probs in flattened_log_probs:
            step_entropies = []
            for log_probs in step_log_probs:
                # Calculate entropy for each token position
                log_probs_array = np.array(log_probs)
                mask = ~np.isinf(log_probs_array)
                if np.any(mask):
                    # Convert log probabilities to probabilities
                    valid_log_probs = log_probs_array[mask]
                    probs = np.exp(valid_log_probs)
                    # Calculate entropy: -sum(p * log(p))
                    entropy = -np.sum(probs * valid_log_probs)
                    step_entropies.append(entropy)
                else:
                    step_entropies.append(0.0)
            steps_entropy.append(step_entropies)

        # Reconstruct the original nested structure
        reconstructed_entropy = reconstruct(steps_entropy, sample_steps_log_probs)

        return {"steps_entropy": reconstructed_entropy}
