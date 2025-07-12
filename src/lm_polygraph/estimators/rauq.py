import numpy as np
from typing import Dict, List, Optional

from .estimator import Estimator


class RAUQ(Estimator):
    """
    RAUQ (Recurrent Attention-based Uncertainty Quantification) from https://arxiv.org/abs/2505.20045

    This estimator quantifies uncertainty in LLM outputs by combining
    attention patterns with token probabilities in a recurrent manner.

    Args:
        alpha: Weight parameter for combining attention and probability scores
        model_name: Name or path of the model to load configuration from
        use_entropy: Whether to use entropy-based uncertainty
        instruct: Whether the model is instruction-tuned
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        use_entropy: bool = False,
        instruct: bool = False,
    ):
        dependencies = ["attention_all", "greedy_log_likelihoods"]
        if use_entropy:
            dependencies.append("entropy")
        super().__init__(dependencies, "sequence")

        self.use_entropy = use_entropy
        self.instruct = instruct
        self.alpha = alpha if alpha is not None else self.get_alpha()

        self.n_layers = n_layers
        self.n_heads = n_heads

        # Focus on middle third of layers which typically contain most relevant information
        if self.n_layers is not None:
            self.layers = list(
                range(self.n_layers // 3, int(np.ceil(self.n_layers / 3 * 2) + 1))
            )
        else:
            self.layers = None

    def __str__(self) -> str:
        """Returns a string representation of the estimator."""
        method_desc = " (entropy)" if self.use_entropy else ""
        return f"RAUQ{method_desc}"

    def get_alpha(self) -> float:
        """
        Returns the default alpha parameter based on model configuration.

        Returns:
            float: Alpha value between 0 and 1
        """
        if self.instruct:
            return 0.9 if self.use_entropy else 0.5
        return 0.8 if self.use_entropy else 0.2

    def _calculate_confidence_scores(
        self,
        log_probabilities: np.ndarray,
        attentions: np.ndarray,
        layer: int,
        head: int,
    ) -> List[float]:
        """
        Calculate confidence scores for a sequence using attention and probabilities.

        Args:
            log_probabilities: Log probabilities for each token
            attentions: Attention patterns
            layer: Current layer index
            head: Selected attention head

        Returns:
            List[float]: Confidence scores for each position
        """
        confidence_scores = [np.exp(log_probabilities[0])]

        for j in range(1, len(log_probabilities)):
            current_prob = np.exp(log_probabilities[j])
            prev_confidence = confidence_scores[-1]

            attention_weight = attentions[layer, head, j - 1]
            confidence = (
                self.alpha * current_prob
                + (1 - self.alpha) * attention_weight * prev_confidence
            )
            confidence_scores.append(confidence)

        return confidence_scores

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty scores for each sequence in a batch.

        Args:
            stats: Dictionary containing model statistics including attention weights and log likelihoods

        Returns:
            np.ndarray: Uncertainty scores for each sequence
        """
        if self.n_layers is None:
            self.n_layers = stats["model"].model.config.num_hidden_layers
            self.layers = list(
                range(self.n_layers // 3, int(np.ceil(self.n_layers / 3 * 2) + 1))
            )
        if self.n_heads is None:
            self.n_heads = stats["model"].model.config.num_attention_heads

        # Extract diagonal attention patterns for each sequence
        attentions = []
        for attention_weight in stats["attention_all"]:
            # Reshape attention weights to separate layers and heads
            reshaped_weights = attention_weight.reshape(
                self.n_layers,
                self.n_heads,
                attention_weight.shape[-2],
                attention_weight.shape[-1],
            )
            # Extract attention weights for previous token with offset -1
            attenion_prev_token = np.diagonal(
                reshaped_weights, offset=-1, axis1=2, axis2=3
            )
            attentions.append(attenion_prev_token)
        greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        if self.use_entropy:
            entropy = stats["entropy"]
            vocab_size = len(stats["greedy_log_probs"][0][0])
            max_entropy = np.log(vocab_size)

        uncertainty_scores = []

        for idx in range(len(greedy_log_likelihoods)):
            # Get log probabilities for current sequence
            if self.use_entropy:
                log_probabilities = np.log(max_entropy - np.array(entropy[idx]) + 1e-10)
            else:
                log_probabilities = greedy_log_likelihoods[idx]

            # Calculate uncertainty scores for each layer
            layer_scores = []
            for layer in self.layers:
                # Select most attentive head for current layer
                head = attentions[idx][layer].mean(-1).argmax()

                # Calculate confidence scores
                confidence_scores = self._calculate_confidence_scores(
                    log_probabilities, attentions[idx], layer, head
                )

                # Calculate uncertainty score
                uncertainty = 1 - np.log(confidence_scores).mean()
                layer_scores.append(uncertainty)

            # Take maximum uncertainty across layers
            uncertainty_scores.append(np.max(layer_scores))

        return np.array(uncertainty_scores)
