import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from lm_polygraph.estimators.estimator import Estimator

class LookBackLens(Estimator):
    """
    This estimator implements the LookBackLens method from 
    "LookBackLens: Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps"
    (https://aclanthology.org/2024.emnlp-main.84/).

    The method uses the model's attention lookback ratios as features for a supervised
    classifier (logistic regression) to predict hallucination uncertainty. For each
    generated sequence, token-level lookback ratios are averaged to form a fixed-size
    feature vector. The classifier is trained to distinguish between hallucinated and
    non-hallucinated outputs based on a metric threshold.

    Args:
        metric_thr (float): Threshold for binarizing the metric (e.g., AlignScore).

    Dependencies:
        - "lookback_ratios"
        - "train_lookback_ratios"
        - "train_metrics"
    """

    def __init__(
        self,
        metric_thr: float = 0.3,
    ):
        super().__init__(
            [
                "lookback_ratios",
                "train_lookback_ratios",
                "train_metrics",
            ],
            "sequence",
        )

        self.classifier = LogisticRegression(max_iter=1000)
        self.metric_thr = metric_thr
        self.is_fitted = False

    def __str__(self):
        return f"LookBackLens"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute LookBackLens uncertainty scores for a batch of samples.

        Args:
            stats (Dict[str, np.ndarray]): Dictionary containing lookback ratios,
                greedy tokens, and train metrics.

        Returns:
            np.ndarray: Array of uncertainty scores (1 - predicted probability of non-hallucination).
        """
        if not self.is_fitted:
            train_lookback_ratios = np.array(stats["train_lookback_ratios"])
            train_greedy_tokens = stats["train_greedy_tokens"]
            train_metrics = stats["train_metrics"]
            targets = (train_metrics > self.metric_thr).astype(int)

            features = []
            k = 0
            for greedy_tokens in train_greedy_tokens:
                features.append(
                    train_lookback_ratios[k : k + len(greedy_tokens)].mean(0)
                )
                k += len(greedy_tokens)
            features = np.array(features)
            self.classifier.fit(features, targets)
            self.is_fitted = True

        lookback_ratios = np.array(stats["lookback_ratios"])
        greedy_tokens = np.array(stats["greedy_tokens"])
        features = []
        k = 0
        for greedy_token in greedy_tokens:
            features.append(lookback_ratios[k : k + len(greedy_token)].mean(0))
            k += len(greedy_token)

        uq = 1 - self.classifier.predict_proba(features)[:, 1]
        return uq
