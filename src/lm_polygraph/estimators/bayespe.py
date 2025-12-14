import numpy as np
import pickle
from typing import List, Dict, Optional

from sklearn.metrics import log_loss

from .estimator import Estimator

SMALL_CONSTANT = 1e-5

def replace_nans_with_uniform(probs):
    for i in range(probs.shape[0]):
        if np.isnan(probs[i, :]).any():
            probs[i, :] = 1.0 / probs.shape[1]
    return probs


def smooth_probs_3d(probs_3d):
    probs_new = probs_3d + SMALL_CONSTANT
    for i in range(probs_new.shape[2]):
        probs_new[:, :, i] = replace_nans_with_uniform(probs_new[:, :, i])
        for j in range(probs_new.shape[0]):
            probs_new[j, :, i] = probs_new[j, :, i] / np.sum(probs_new[j, :, i])
    return probs_new


class BayesPEZeroShot(Estimator):
    """
    Bayesian Prompt Ensembles for Zero-Shot classification (BPE-style).
    """
    def __init__(
        self,
        instructions: List[str],
        class_labels: Optional[List[str]] = None,
        prompt_formatting: Optional[str] = None,
        n_iterations_weights_optimiser: int = 10,
    ):
        super().__init__(["ensemble_probs"], "sequence")
        self.instructions = instructions
        self.class_labels = class_labels
        self.prompt_formatting = prompt_formatting
        self.n_iterations_weights_optimiser = n_iterations_weights_optimiser
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return f"BayesPEZeroShot({len(self.instructions)} prompts)"

    def optimise_weights(self, val_probs_ensemble: np.ndarray, val_labels: np.ndarray, learning_rate=SMALL_CONSTANT):
        # val_probs_ensemble: [n_samples, n_classes, n_instructions]
        probs = smooth_probs_3d(val_probs_ensemble)
        nan_cost = True
        lr = learning_rate
        weights = np.ones(probs.shape[2]) / probs.shape[2]
        for _ in range(self.n_iterations_weights_optimiser):
            # Weighted sum over instructions
            ensemble_probs = np.tensordot(probs, weights, axes=([2], [0]))
            ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)
            try:
                cost = log_loss(val_labels, ensemble_probs)
            except Exception:
                cost = np.nan
            if not np.isnan(cost):
                break
            lr *= 0.5
        self.weights = weights
        return weights

    def forward(self, test_probs_ensemble: np.ndarray, n_forward_passes: Optional[int] = None):
        # test_probs_ensemble: [n_samples, n_classes, n_instructions]
        if n_forward_passes is None:
            n_forward_passes = len(self.instructions)
        chosen_indices = np.argsort(self.weights)[-n_forward_passes:]
        chosen_weights = self.weights[chosen_indices]
        chosen_weights = chosen_weights / np.sum(chosen_weights)
        probs = test_probs_ensemble[:, :, chosen_indices]
        probs = smooth_probs_3d(probs)
        # Weighted sum
        ensemble_probs = np.tensordot(probs, chosen_weights, axes=([2], [0]))
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)
        return ensemble_probs

    def save_weights(self, save_dir="saved_weights/ensemble_weights"):
        with open(save_dir, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, load_dir="saved_weights/ensemble_weights"):
        with open(load_dir, "rb") as f:
            self.weights = pickle.load(f)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # stats["ensemble_probs"]: [n_samples, n_classes, n_instructions]
        probs_ensemble = stats["ensemble_probs"]
        ensemble_probs = self.forward(probs_ensemble)
        # Uncertainty as entropy
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
        return entropy


class BayesPEFewShot(Estimator):
    """
    Bayesian Prompt Ensembles for Few-Shot classification (BPE-style).
    """
    def __init__(
        self,
        instructions: List[str],
        few_shot_examples: List[Dict[str, str]],
        class_labels: Optional[List[str]] = None,
        prompt_formatting: Optional[str] = None,
        n_iterations_weights_optimiser: int = 10,
    ):
        super().__init__(["ensemble_probs"], "sequence")
        self.instructions = instructions
        self.few_shot_examples = few_shot_examples
        self.class_labels = class_labels
        self.prompt_formatting = prompt_formatting
        self.n_iterations_weights_optimiser = n_iterations_weights_optimiser
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return f"BayesPEFewShot({len(self.instructions)} prompts, {len(self.few_shot_examples)} examples)"

    def optimise_weights(self, val_probs_ensemble: np.ndarray, val_labels: np.ndarray, learning_rate=SMALL_CONSTANT):
        probs = smooth_probs_3d(val_probs_ensemble)
        nan_cost = True
        lr = learning_rate
        weights = np.ones(probs.shape[2]) / probs.shape[2]
        for _ in range(self.n_iterations_weights_optimiser):
            ensemble_probs = np.tensordot(probs, weights, axes=([2], [0]))
            ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)
            try:
                cost = log_loss(val_labels, ensemble_probs)
            except Exception:
                cost = np.nan
            if not np.isnan(cost):
                break
            lr *= 0.5
        self.weights = weights
        return weights

    def forward(self, test_probs_ensemble: np.ndarray, n_forward_passes: Optional[int] = None):
        if n_forward_passes is None:
            n_forward_passes = len(self.instructions)
        chosen_indices = np.argsort(self.weights)[-n_forward_passes:]
        chosen_weights = self.weights[chosen_indices]
        chosen_weights = chosen_weights / np.sum(chosen_weights)
        probs = test_probs_ensemble[:, :, chosen_indices]
        probs = smooth_probs_3d(probs)
        ensemble_probs = np.tensordot(probs, chosen_weights, axes=([2], [0]))
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)
        return ensemble_probs

    def save_weights(self, save_dir="saved_weights/ensemble_weights"):
        with open(save_dir, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, load_dir="saved_weights/ensemble_weights"):
        with open(load_dir, "rb") as f:
            self.weights = pickle.load(f)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        probs_ensemble = stats["ensemble_probs"]
        ensemble_probs = self.forward(probs_ensemble)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
        return entropy
