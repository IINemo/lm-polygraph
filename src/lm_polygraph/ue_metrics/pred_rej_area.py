import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import uuid

class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __str__(self):
        return "prr"

    def get_ue_rejection(self, estimator, target, num_remaining_points):
        return np.flip(np.cumsum(target[np.argsort(estimator)]) / num_remaining_points)

    def get_oracle_rejection(self, estimator, target, num_remaining_points):
        return np.flip(np.cumsum(np.flip(np.sort(target))) / num_remaining_points)

    def get_random_rejection(self, estimator, target, num_remaining_points, N_EXAMPLES):
        random_rejection_accuracies = []
        for _ in range(1000):
            order = np.arange(0, N_EXAMPLES)
            np.random.shuffle(order)
            random_rejection_accuracies.append(np.flip(np.cumsum(target[order]) / num_remaining_points))

        return np.mean(random_rejection_accuracies, axis=0)

def __call__(self, estimator: List[float], target: List[float], generate_curve:bool = False, e_level:str = '', e_name:str ='', gen_name:str ='', ue_metric:str ='') -> float:
    """
    Measures the area under the Prediction-Rejection curve between `estimator` and `target`.

    Parameters:
        estimator (List[float]): A batch of uncertainty estimations.
            Higher values indicate more uncertainty.
        target (List[float]): A batch of ground-truth uncertainty estimations.
            Higher values indicate less uncertainty.
        generate_curve (bool): A flag to generate and save the PRR curve if set to True.
        e_level (str): Name of method level.
        e_name (str): Name of estimattor method.
        gen_name (str): Name of generation metric.
        ue_metric (str): The uncertainty estimation metric used (for labeling purposes).

    Returns:
        float: Area under the Prediction-Rejection curve (PRR score).
            Higher values indicate better uncertainty estimations.
    """
    # Normalize the target values to a common scale
    target = normalize(target)
    
    # Convert the estimator list to a NumPy array (UE stands for uncertainty estimation)
    ue = np.array(estimator)
    num_obs = len(ue)
    
    # Sort the indices of `ue` in ascending order, so least uncertain examples come first
    ue_argsort = np.argsort(ue)
    
    # Sort the target metrics based on the sorted indices from the estimator
    sorted_metrics = np.array(target)[ue_argsort]
    
    # Compute the cumulative sum of the sorted metrics for calculating the PRR score
    cumsum = np.cumsum(sorted_metrics)
    
    # Calculate the scores as cumulative sums divided by the index (from the sorted order)
    # and reverse the scores to get the final rejection curve
    scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
    
    # The PRR score is the average of the reversed cumulative sums, divided by the number of observations
    prr_score = np.sum(scores) / num_obs

    # If `generate_curve` is set to True, generate and save the PRR curve plot
    if generate_curve:
        plots_dir = './plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Get the number of examples and remaining points to calculate rejection accuracies
        N_EXAMPLES = len(estimator)
        num_remaining_points = np.arange(1, N_EXAMPLES + 1)

        # Calculate rejection accuracies for the UE (uncertainty estimator), Oracle, and Random baselines
        ue_rejected_accuracy = self.get_ue_rejection(estimator, target, num_remaining_points)
        oracle_rejected_accuracy = self.get_oracle_rejection(estimator, target, num_remaining_points)
        random_rejection_accuracy = self.get_random_rejection(estimator, target, num_remaining_points, N_EXAMPLES)

        # Define the rejection rates, ranging from 0 (keeping all data) to 1 (discarding all data)
        rejection_rates = np.linspace(0, 1, N_EXAMPLES)

        # Plot the rejection curves for UE, Oracle, and Random using Seaborn for better visualization
        sns.lineplot(x=rejection_rates, y=ue_rejected_accuracy, label='UE')
        sns.lineplot(x=rejection_rates, y=oracle_rejected_accuracy, label='Oracle')
        g = sns.lineplot(x=rejection_rates, y=random_rejection_accuracy, label='Random')
        
        # Customize the plot's labels, title, and grid for better readability
        g.set_xlabel('Rejection Rate')
        g.set_ylabel(f'{gen_name}')
        g.set_title(f'PRR curve: {e_level}, {e_name}')
        g.grid()
        
        # Generate a unique filename for the plot using UUID and save it as a PNG file
        base_filename = 'prr_curve'
        extension = 'png'
        unique_id = uuid.uuid4()
        new_filename = f"{base_filename}_{e_name}_{gen_name}_{unique_id}.{extension}"
        save_path = os.path.join(plots_dir, new_filename)

        # Save the generated plot and close the figure to free up memory
        plt.savefig(save_path)
        plt.close() 

    # Return the computed PRR score
    return prr_score
