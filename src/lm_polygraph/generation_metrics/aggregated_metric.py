import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AggregatedMetric(GenerationMetric):
    """
    Aggregated metric class, which wraps a base metric and aggregates its results for multi-target datasets.
    """

    def __init__(self, base_metric: GenerationMetric, aggregation: str = "max"):
        self.base_metric = base_metric
        self.level = base_metric.level
        self.stats_dependencies = base_metric.stats_dependencies
        self.aggregation = aggregation

    def __str__(self):
        # Check if base_metric is a wrapper, else return base_metric
        return str(getattr(self.base_metric, "base_metric", self.base_metric))

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates aggregated metric between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): calculated stats
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of aggregated metric values for each sample in input.
        """
        metric_values = []
        for i, (targets, greedy_text) in enumerate(
            zip(target_texts, stats["greedy_texts"])
        ):
            # truncate stats to only process one sample at a time
            truncated_stats = {
                k: [v[i]]
                for k, v in stats.items()
                if k in self.stats_dependencies + ["greedy_texts"]
            }

            sample_metric_values = []
            for target in targets:
                value = self.base_metric(truncated_stats, [target])
                sample_metric_values.append(value)

            if self.aggregation == "max":
                metric_values.append(np.nanmax(sample_metric_values))
            else:
                raise ValueError(f"Unknown aggregation type: {self.aggregation}")

        return np.array(metric_values)
