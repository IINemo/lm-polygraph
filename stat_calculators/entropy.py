import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict, List

from stat_calculators.stat_calculator import StatCalculator
from utils.model import Model


class EntropyCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['entropy'], ['greedy_log_probs'])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        logprobs = dependencies['greedy_log_probs']
        entropies = []
        for s_lp in logprobs:
            entropies.append([])
            for lp in s_lp:
                entropies[-1].append(-np.mean(np.array(lp) * np.exp(lp)))
        return {'entropy': entropies}
