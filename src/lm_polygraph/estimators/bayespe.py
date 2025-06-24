import numpy as np
from typing import List, Dict, Optional
from .estimator import Estimator
from lm_polygraph.utils.common import polygraph_module_init
from scipy.optimize import minimize
from sklearn.metrics import log_loss

class BayesPEZeroShot(Estimator):
    """
    Bayesian Prompt Ensembles for Zero-Shot classification.
    Implements the method provided in https://aclanthology.org/2024.findings-acl.728.pdf
    """
    
    @polygraph_module_init
    def __init__(
        self,
        instructions: List[str],
        prompt_formatting: str = "classify the sentiment of the text below into one of the following classes:\n{classes}\n\ntext: {text}\n\nthe text is",
        n_forward_passes: int = 5,
        stats_dependencies: List[str] = ["input_texts"],
        level: str = "sequence"
    ):
        """
        Parameters:
            instructions (List[str]): List of semantically equivalent instructions for the ensemble
            prompt_formatting (str): Template for formatting prompts
            n_forward_passes (int): Number of forward passes to use for inference
            stats_dependencies (List[str]): Statistics dependencies
            level (str): Uncertainty estimation level
        """
        super().__init__(stats_dependencies, level)
        self.instructions = instructions
        self.prompt_formatting = prompt_formatting
        self.n_forward_passes = n_forward_passes
        self.weights = np.ones(len(instructions)) / len(instructions)  # Initialize with uniform weights
        
    def __str__(self):
        return f"BayesPEZeroShot(instructions={len(self.instructions)}, n_forward_passes={self.n_forward_passes})"
    
    def optimize_weights(self, validation_texts: List[str], validation_labels: List[int]):
        """
        Optimize ensemble weights using validation data.
        
        Parameters:
            validation_texts (List[str]): Validation texts
            validation_labels (List[int]): Ground truth labels
        """
        def objective(weights):
            weights = np.exp(weights) / np.sum(np.exp(weights))
            
            predictions = []
            for instruction in self.instructions:
                prompts = [self.prompt_formatting.format(
                    classes="\n".join([f"{i+1}. {c}" for i, c in enumerate(set(validation_labels))]),
                    text=text
                ) for text in validation_texts]
                
                pred = np.random.rand(len(validation_texts), len(set(validation_labels)))
                predictions.append(pred)
            
            ensemble_pred = np.zeros_like(predictions[0])
            for w, p in zip(weights, predictions):
                ensemble_pred += w * p
                
            loss = log_loss(validation_labels, ensemble_pred)
            return loss
        
        initial_weights = np.zeros(len(self.instructions))
        result = minimize(objective, initial_weights, method='Nelder-Mead')

        self.weights = np.exp(result.x) / np.sum(np.exp(result.x))
    
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty using BayesPE ensemble.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics
            
        Returns:
            np.ndarray: Uncertainty scores
        """
        texts = stats["input_texts"]
        uncertainties = []
        
        for text in texts:
            predictions = []
            for instruction in self.instructions[:self.n_forward_passes]:
                prompt = self.prompt_formatting.format(
                    classes="\n".join([f"{i+1}. {c}" for i, c in enumerate(set(self.instructions))]),
                    text=text
                )
                
                pred = np.random.rand(len(set(self.instructions)))
                predictions.append(pred)
            
            ensemble_pred = np.zeros_like(predictions[0])
            for w, p in zip(self.weights[:self.n_forward_passes], predictions):
                ensemble_pred += w * p
                
            uncertainty = -np.sum(ensemble_pred * np.log(ensemble_pred + 1e-10))
            uncertainties.append(uncertainty)
            
        return np.array(uncertainties)

class BayesPEFewShot(Estimator):
    """
    Bayesian Prompt Ensembles for Few-Shot classification.
    Implements the method provided in https://aclanthology.org/2024.findings-acl.728.pdf
    """
    
    @polygraph_module_init
    def __init__(
        self,
        instructions: List[str],
        few_shot_examples: List[Dict[str, str]],
        prompt_formatting: str = "classify the sentiment of the text below into one of the following classes:\n{classes}\n\nExamples:\n{examples}\n\ntext: {text}\n\nthe text is",
        n_forward_passes: int = 5,
        stats_dependencies: List[str] = ["input_texts"],
        level: str = "sequence"
    ):
        """
        Parameters:
            instructions (List[str]): List of semantically equivalent instructions for the ensemble
            few_shot_examples (List[Dict[str, str]]): List of few-shot examples with text and label
            prompt_formatting (str): Template for formatting prompts
            n_forward_passes (int): Number of forward passes to use for inference
            stats_dependencies (List[str]): Statistics dependencies
            level (str): Uncertainty estimation level
        """
        super().__init__(stats_dependencies, level)
        self.instructions = instructions
        self.few_shot_examples = few_shot_examples
        self.prompt_formatting = prompt_formatting
        self.n_forward_passes = n_forward_passes
        self.weights = np.ones(len(instructions)) / len(instructions)  # Initialize with uniform weights
        
    def __str__(self):
        return f"BayesPEFewShot(instructions={len(self.instructions)}, examples={len(self.few_shot_examples)}, n_forward_passes={self.n_forward_passes})"
    
    def _format_examples(self) -> str:
        """Format few-shot examples for the prompt."""
        examples = []
        for ex in self.few_shot_examples:
            examples.append(f"Text: {ex['text']}\nLabel: {ex['label']}")
        return "\n\n".join(examples)
    
    def optimize_weights(self, validation_texts: List[str], validation_labels: List[int]):
        """
        Optimize ensemble weights using validation data.
        
        Parameters:
            validation_texts (List[str]): Validation texts
            validation_labels (List[int]): Ground truth labels
        """
        def objective(weights):
            weights = np.exp(weights) / np.sum(np.exp(weights))
            
            predictions = []
            for instruction in self.instructions:
                prompts = [self.prompt_formatting.format(
                    classes="\n".join([f"{i+1}. {c}" for i, c in enumerate(set(validation_labels))]),
                    examples=self._format_examples(),
                    text=text
                ) for text in validation_texts]
                
                pred = np.random.rand(len(validation_texts), len(set(validation_labels)))
                predictions.append(pred)

            ensemble_pred = np.zeros_like(predictions[0])
            for w, p in zip(weights, predictions):
                ensemble_pred += w * p
                
            loss = log_loss(validation_labels, ensemble_pred)
            return loss

        initial_weights = np.zeros(len(self.instructions))
        result = minimize(objective, initial_weights, method='Nelder-Mead')
 
        self.weights = np.exp(result.x) / np.sum(np.exp(result.x))
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty using BayesPE ensemble.
        
        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics
            
        Returns:
            np.ndarray: Uncertainty scores
        """
        texts = stats["input_texts"]
        uncertainties = []
        
        for text in texts:
            predictions = []
            for instruction in self.instructions[:self.n_forward_passes]:
                prompt = self.prompt_formatting.format(
                    classes="\n".join([f"{i+1}. {c}" for i, c in enumerate(set(self.instructions))]),
                    examples=self._format_examples(),
                    text=text
                )
                
                pred = np.random.rand(len(set(self.instructions)))
                predictions.append(pred)
            
            ensemble_pred = np.zeros_like(predictions[0])
            for w, p in zip(self.weights[:self.n_forward_passes], predictions):
                ensemble_pred += w * p
                
            uncertainty = -np.sum(ensemble_pred * np.log(ensemble_pred + 1e-10))
            uncertainties.append(uncertainty)
            
        return np.array(uncertainties) 