import numpy as np
from typing import List, Dict, Optional
from .estimator import Estimator
from lm_polygraph.utils.common import polygraph_module_init
from scipy.optimize import minimize
from sklearn.metrics import log_loss

class BayesPEZeroShot(Estimator):
    """
    Bayesian Prompt Ensembles for Zero-Shot classification.
    Estimates uncertainty by aggregating log-probabilities from multiple prompts (instructions).
    """
    def __init__(self, instructions: List[str], n_forward_passes: int = 5):
        super().__init__(["greedy_log_probs"], "sequence")
        self.instructions = instructions
        self.n_forward_passes = n_forward_passes
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return f"BayesPEZeroShot({len(self.instructions)} prompts, n_forward_passes={self.n_forward_passes})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates uncertainty as the entropy of the weighted average probability distribution
        over the ensemble of prompts (instructions).
        Expects stats["greedy_log_probs"] to be a list of lists of log-probabilities for each prompt.
        """
        
        log_probs_ensemble = stats["greedy_log_probs"]  
        log_probs_ensemble = np.array(log_probs_ensemble) 

        seq_log_probs = np.sum(log_probs_ensemble, axis=2)  

        seq_probs = np.exp(seq_log_probs)

        weighted_probs = np.average(seq_probs, axis=0, weights=self.weights[:log_probs_ensemble.shape[0]])

        uncertainties = -np.log(weighted_probs + 1e-10)
        return uncertainties

class BayesPEFewShot(Estimator):
    """
    Bayesian Prompt Ensembles for Few-Shot classification.
    Estimates uncertainty by aggregating log-probabilities from multiple prompts (instructions) with few-shot examples.
    """
    def __init__(self, instructions: List[str], n_forward_passes: int = 5):
        super().__init__(["greedy_log_probs"], "sequence")
        self.instructions = instructions
        self.n_forward_passes = n_forward_passes
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return f"BayesPEFewShot({len(self.instructions)} prompts, n_forward_passes={self.n_forward_passes})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        
        log_probs_ensemble = stats["greedy_log_probs"] 
        log_probs_ensemble = np.array(log_probs_ensemble)
        seq_log_probs = np.sum(log_probs_ensemble, axis=2)
        seq_probs = np.exp(seq_log_probs)
        weighted_probs = np.average(seq_probs, axis=0, weights=self.weights[:log_probs_ensemble.shape[0]])
        uncertainties = -np.log(weighted_probs + 1e-10)
        return uncertainties 