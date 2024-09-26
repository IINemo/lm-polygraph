import numpy as np

from typing import List
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import os
import uuid

def normalize(target: List[float]):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


def skip_target_nans(target, estimator):
    newt, newe = [], []
    for t, e in zip(target, estimator):
        if np.isnan(t):
            continue
        newt.append(t)
        newe.append(e)
    return newt, newe


class UEMetric(ABC):
    """
    Abstract class, which measures the quality of uncertainty estimations from some Estimator using
    ground-truth uncertainty estimations calculated from some GenerationMetric.
    """

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the UEMetric.
        Class parameters which affect generation metric estimates should also be included in the unique name
        to diversify between UEMetric's.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Abstract method. Measures the quality of uncertainty estimations `estimator`
        by comparing them to the ground-truth uncertainty estimations `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: a quality measure of `estimator` estimations.
                Higher values can indicate either better or lower qualities,
                which depends on a particular implementation.
        """
        raise Exception("Not implemented")



def get_random_scores(function, metrics, return_scores:bool = False, num_iter=1000, seed=42):
    np.random.seed(seed)
   
    rand_scores = np.arange(len(metrics))

    prr_values = []      # To store PRR scores across iterations
    score_values = []    # To store detailed scores or rejection accuracies across iterations

    for _ in range(num_iter):
        np.random.shuffle(rand_scores)

        # Use the function like __call__ to compute PRR score and optionally return detailed scores
        if return_scores:
            # Call the function to get both PRR score and detailed scores
            prr_score, detailed_scores = function(rand_scores, metrics, return_scores=True)
            prr_values.append(prr_score)
            score_values.append(detailed_scores)
        else:
            # Call the function to get only the PRR score
            prr_score = function(rand_scores, metrics)
            prr_values.append(prr_score)

    # Compute the mean PRR score across all iterations
    mean_prr_score = np.mean(prr_values)

    # If return_scores is True, also compute the mean of the detailed scores (or rejection accuracies)
    if return_scores:
        mean_scores = np.mean(score_values, axis=0)
        return mean_prr_score, mean_scores

    # Otherwise, just return the PRR score (no detailed scores)
    return mean_prr_score



def normalize_metric(target_score, oracle_score, random_score):
    if not (oracle_score == random_score):
        target_score = (target_score - random_score) / (oracle_score - random_score)
    return target_score




def generate_prr_curve(ue_rejected_accuracy, oracle_rejected_accuracy, random_rejected_accuracy, e_level: str, e_name: str, gen_name: str):
    """
    Generates and saves a PRR curve plot using only matplotlib.

    Parameters:
        ue_rejected_accuracy (np.array): Rejection curve for uncertainty estimation (UE).
        oracle_rejected_accuracy (np.array): Rejection curve for Oracle (ideal).
        random_rejected_accuracy (np.array): Rejection curve for Random baseline.
        e_level (str): Experiment level.
        e_name (str): Experiment name.
        gen_name (str): General name for the plot label.

    Returns:
        str: The path where the plot is saved.
    """
    # Directory to save plots
    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Number of examples
    N_EXAMPLES = len(ue_rejected_accuracy)
    
    # Rejection rates (x-axis)
    rejection_rates = np.linspace(0, 1, N_EXAMPLES)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Plot each line (UE, Oracle, Random)
    plt.plot(rejection_rates, ue_rejected_accuracy, label='UE', linestyle='-')
    plt.plot(rejection_rates, oracle_rejected_accuracy, label='Oracle', linestyle='-')
    plt.plot(rejection_rates, random_rejected_accuracy, label='Random', linestyle='-')

    # Add labels and title
    plt.xlabel('Rejection Rate')
    plt.ylabel(f'{gen_name}')
    plt.title(f'PRR curve: {e_level}, {e_name}')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Generate a random UUID for the filename
    base_filename = 'prr_curve'
    extension = 'png'
    unique_id = uuid.uuid4()
    new_filename = f"{base_filename}_{e_name}_{gen_name}_{unique_id}.{extension}"
    save_path = os.path.join(plots_dir, new_filename)

    # Save the plot
    plt.savefig(save_path)
    plt.close()