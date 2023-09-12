import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class StdCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['std'], ['greedy_log_probs'])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, max_new_tokens: int = 100) -> Dict[str, np.ndarray]:
        logprobs = dependencies['greedy_log_probs']
        stds = []
        for s_lp in logprobs:
            stds.append([])
            for lp in s_lp:
                mask = ~np.isinf(lp)
                logp = np.array(lp[mask])
                stds[-1].append(np.std(logp))
        return {'std': stds}
