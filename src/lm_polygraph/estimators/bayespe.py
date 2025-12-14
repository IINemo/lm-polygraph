"""
Bayesian Prompt Ensembles (BayesPE) for uncertainty estimation.

Implements the method from:
"Bayesian Prompt Ensembles: Model Uncertainty Estimation for Black-Box Large Language Models"
https://aclanthology.org/2024.findings-acl.728.pdf
"""

import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .estimator import Estimator

SMALL_CONSTANT = 1e-5


def replace_nans_with_uniform(probs: np.ndarray) -> np.ndarray:
    """
    Replace probability distributions containing NaNs with uniform distributions.

    Args:
        probs: 2D array of probability distributions [n_samples, n_classes]

    Returns:
        2D array with NaN rows replaced by uniform distributions.
    """
    for i in range(probs.shape[0]):
        if np.isnan(probs[i, :]).any():
            probs[i, :] = 1.0 / probs.shape[1]
    return probs


def smooth_probs_3d(probs_3d: np.ndarray) -> np.ndarray:
    """
    Smooth and normalize 3D probability arrays to avoid numerical issues.

    Args:
        probs_3d: 3D array [n_samples, n_classes, n_instructions]

    Returns:
        Smoothed and normalized 3D array.
    """
    probs_new = probs_3d + SMALL_CONSTANT
    for i in range(probs_new.shape[2]):
        probs_new[:, :, i] = replace_nans_with_uniform(probs_new[:, :, i])
        for j in range(probs_new.shape[0]):
            probs_new[j, :, i] = probs_new[j, :, i] / np.sum(probs_new[j, :, i])
    return probs_new


class BayesianLoss(nn.Module):
    """
    BayesPE loss function (Equation 4 in the paper).

    Combines negative log-likelihood with an entropy regularization term
    to encourage diverse prompt weighting.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        scales_logits: torch.Tensor,
        probs_y: torch.Tensor,
        kl_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute the BayesPE loss.

        Args:
            scales_logits: Logits for prompt weights (log(w)) [n_instructions]
            probs_y: Probabilities for ground-truth labels per instruction
                     [n_validation_examples, n_instructions]
            kl_weight: Weight for entropy regularization term.
                       Higher values encourage more uniform weights.

        Returns:
            BayesPE loss value.
        """
        # Softmax to get normalized weights
        scales = torch.exp(scales_logits) / (
            torch.sum(torch.exp(scales_logits)) + SMALL_CONSTANT
        )
        # Negative log-likelihood of ensemble predictions
        likelihood = torch.sum(
            torch.log(torch.matmul(probs_y.double(), scales.double()) + SMALL_CONSTANT)
        )
        # Entropy regularization (encourages weight diversity)
        entropy = -torch.dot(
            scales.double(), torch.log(scales.double() + SMALL_CONSTANT)
        )
        return -likelihood - kl_weight * entropy


class EnsembleScaler:
    """
    Optimizer for BayesPE prompt weights using LBFGS.
    """

    def __init__(self, n_iterations: int = 10, verbose: bool = False):
        """
        Args:
            n_iterations: Number of optimization iterations.
            verbose: Whether to print progress during optimization.
        """
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.criterion = BayesianLoss()

    @staticmethod
    def extract_probs_y(probs_pred: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
        """
        Extract probabilities assigned to ground-truth labels.

        Args:
            probs_pred: 3D array [n_samples, n_classes, n_instructions]
            gt_labels: 1D array of ground-truth label indices [n_samples]

        Returns:
            2D array [n_samples, n_instructions] of probabilities for GT labels.
        """
        n_samples = probs_pred.shape[0]
        n_instructions = probs_pred.shape[2]
        probs = np.zeros((n_samples, n_instructions))
        for i in range(n_samples):
            for j in range(n_instructions):
                probs[i, j] = probs_pred[i, int(gt_labels[i]), j] + SMALL_CONSTANT
        return probs

    def train(
        self,
        probs_train: np.ndarray,
        gt_labels: np.ndarray,
        lr: float = 0.001,
        kl_weight: float = 0.1,
    ) -> tuple:
        """
        Train prompt weights using LBFGS optimization.

        Args:
            probs_train: 3D array [n_samples, n_classes, n_instructions]
            gt_labels: 1D array of ground-truth labels [n_samples]
            lr: Learning rate for LBFGS optimizer.
            kl_weight: Weight for entropy regularization.

        Returns:
            Tuple of (optimized weights, list of loss values).
        """
        # Determine device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract probabilities for ground-truth labels
        probs_y_np = self.extract_probs_y(probs_train, gt_labels)
        probs_y = torch.from_numpy(probs_y_np).to(device)

        # Initialize weight logits
        n_instructions = probs_train.shape[2]
        scales = nn.Parameter(torch.ones(n_instructions, device=device))

        # LBFGS optimizer (as in original BayesPE paper)
        optimizer = optim.LBFGS(
            [scales], lr=lr, max_iter=100, line_search_fn="strong_wolfe"
        )

        scales_history = []
        losses = []

        def closure():
            optimizer.zero_grad()
            loss = self.criterion(scales, probs_y, kl_weight=kl_weight)
            loss.backward()
            scales_history.append(scales.detach().clone())
            losses.append(float(loss.detach().cpu()))
            return loss

        for i in range(self.n_iterations):
            optimizer.step(closure)
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}, loss: {losses[-1]:.6f}")

        # Convert final logits to normalized weights
        final_scales = scales_history[-1].cpu().numpy()
        p_unnorm = np.exp(final_scales)
        weights = p_unnorm / np.sum(p_unnorm)

        return weights, losses


class BayesPEZeroShot(Estimator):
    """
    Bayesian Prompt Ensembles for Zero-Shot classification.

    Uses multiple semantically equivalent prompts and learns optimal weights
    to combine their predictions for improved calibration.
    """

    def __init__(
        self,
        instructions: List[str],
        class_labels: Optional[List[str]] = None,
        prompt_formatting: Optional[str] = None,
        n_iterations_weights_optimiser: int = 10,
        kl_weight: float = 0.1,
    ):
        """
        Args:
            instructions: List of semantically equivalent instruction prompts.
            class_labels: List of class label strings (e.g., ["positive", "negative"]).
            prompt_formatting: Optional custom prompt template.
            n_iterations_weights_optimiser: Number of LBFGS iterations for weight optimization.
            kl_weight: Weight for entropy regularization in BayesPE loss.
        """
        super().__init__(["ensemble_probs"], "sequence")
        self.instructions = instructions
        self.class_labels = class_labels
        self.prompt_formatting = prompt_formatting
        self.n_iterations_weights_optimiser = n_iterations_weights_optimiser
        self.kl_weight = kl_weight
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return f"BayesPEZeroShot({len(self.instructions)} prompts)"

    def optimise_weights(
        self,
        val_probs_ensemble: np.ndarray,
        val_labels: np.ndarray,
        learning_rate: float = 0.001,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Optimize prompt weights using validation data.

        Args:
            val_probs_ensemble: 3D array [n_samples, n_classes, n_instructions]
                containing class probabilities for each prompt.
            val_labels: 1D array of ground-truth label indices.
            learning_rate: Learning rate for LBFGS optimizer.
            verbose: Whether to print optimization progress.

        Returns:
            Optimized weights array.
        """
        probs = smooth_probs_3d(val_probs_ensemble.copy())
        scaler = EnsembleScaler(
            n_iterations=self.n_iterations_weights_optimiser,
            verbose=verbose,
        )
        weights, _ = scaler.train(
            probs, val_labels, lr=learning_rate, kl_weight=self.kl_weight
        )
        self.weights = weights
        return weights

    def forward(
        self, test_probs_ensemble: np.ndarray, n_forward_passes: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute ensemble predictions using weighted combination.

        Args:
            test_probs_ensemble: 3D array [n_samples, n_classes, n_instructions]
            n_forward_passes: Number of top-weighted prompts to use.
                              If None, uses all prompts.

        Returns:
            2D array [n_samples, n_classes] of ensemble predictions.
        """
        if n_forward_passes is None:
            n_forward_passes = len(self.instructions)
        chosen_indices = np.argsort(self.weights)[-n_forward_passes:]
        chosen_weights = self.weights[chosen_indices]
        chosen_weights = chosen_weights / np.sum(chosen_weights)
        probs = test_probs_ensemble[:, :, chosen_indices]
        probs = smooth_probs_3d(probs)
        # Weighted sum over instructions
        ensemble_probs = np.tensordot(probs, chosen_weights, axes=([2], [0]))
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)
        return ensemble_probs

    def save_weights(self, save_path: str = "saved_weights/ensemble_weights"):
        """Save optimized weights to file."""
        with open(save_path, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, load_path: str = "saved_weights/ensemble_weights"):
        """Load weights from file."""
        with open(load_path, "rb") as f:
            self.weights = pickle.load(f)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute uncertainty scores from ensemble probabilities.

        Args:
            stats: Dictionary containing "ensemble_probs" key with
                   3D array [n_samples, n_classes, n_instructions].

        Returns:
            1D array of entropy-based uncertainty scores.
        """
        probs_ensemble = stats["ensemble_probs"]
        ensemble_probs = self.forward(probs_ensemble)
        # Uncertainty as entropy of ensemble distribution
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
        return entropy


class BayesPEFewShot(Estimator):
    """
    Bayesian Prompt Ensembles for Few-Shot classification.

    Extends BayesPEZeroShot with in-context learning examples.
    """

    def __init__(
        self,
        instructions: List[str],
        few_shot_examples: List[Dict[str, str]],
        class_labels: Optional[List[str]] = None,
        prompt_formatting: Optional[str] = None,
        n_iterations_weights_optimiser: int = 10,
        kl_weight: float = 0.1,
    ):
        """
        Args:
            instructions: List of semantically equivalent instruction prompts.
            few_shot_examples: List of dicts with "text" and "label" keys
                               for in-context examples.
            class_labels: List of class label strings.
            prompt_formatting: Optional custom prompt template.
            n_iterations_weights_optimiser: Number of LBFGS iterations.
            kl_weight: Weight for entropy regularization.
        """
        super().__init__(["ensemble_probs"], "sequence")
        self.instructions = instructions
        self.few_shot_examples = few_shot_examples
        self.class_labels = class_labels
        self.prompt_formatting = prompt_formatting
        self.n_iterations_weights_optimiser = n_iterations_weights_optimiser
        self.kl_weight = kl_weight
        self.weights = np.ones(len(instructions)) / len(instructions)

    def __str__(self):
        return (
            f"BayesPEFewShot({len(self.instructions)} prompts, "
            f"{len(self.few_shot_examples)} examples)"
        )

    def optimise_weights(
        self,
        val_probs_ensemble: np.ndarray,
        val_labels: np.ndarray,
        learning_rate: float = 0.001,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Optimize prompt weights using validation data.

        Args:
            val_probs_ensemble: 3D array [n_samples, n_classes, n_instructions]
            val_labels: 1D array of ground-truth label indices.
            learning_rate: Learning rate for LBFGS optimizer.
            verbose: Whether to print optimization progress.

        Returns:
            Optimized weights array.
        """
        probs = smooth_probs_3d(val_probs_ensemble.copy())
        scaler = EnsembleScaler(
            n_iterations=self.n_iterations_weights_optimiser,
            verbose=verbose,
        )
        weights, _ = scaler.train(
            probs, val_labels, lr=learning_rate, kl_weight=self.kl_weight
        )
        self.weights = weights
        return weights

    def forward(
        self, test_probs_ensemble: np.ndarray, n_forward_passes: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute ensemble predictions using weighted combination.

        Args:
            test_probs_ensemble: 3D array [n_samples, n_classes, n_instructions]
            n_forward_passes: Number of top-weighted prompts to use.

        Returns:
            2D array [n_samples, n_classes] of ensemble predictions.
        """
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

    def save_weights(self, save_path: str = "saved_weights/ensemble_weights"):
        """Save optimized weights to file."""
        with open(save_path, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, load_path: str = "saved_weights/ensemble_weights"):
        """Load weights from file."""
        with open(load_path, "rb") as f:
            self.weights = pickle.load(f)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute uncertainty scores from ensemble probabilities.

        Args:
            stats: Dictionary containing "ensemble_probs" key.

        Returns:
            1D array of entropy-based uncertainty scores.
        """
        probs_ensemble = stats["ensemble_probs"]
        ensemble_probs = self.forward(probs_ensemble)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
        return entropy
