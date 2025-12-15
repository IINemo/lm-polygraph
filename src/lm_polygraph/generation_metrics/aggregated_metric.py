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
            target_texts (List[str] or List[List[str]]): ground-truth texts
                When multiref is true, this is List[List[str]] where each inner list
                contains multiple reference answers for one sample.
        Returns:
            np.ndarray: list of aggregated metric values for each sample in input.
        """
        metric_values = []
        for i, (targets, greedy_text) in enumerate(
            zip(target_texts, stats["greedy_texts"])
        ):
            # Ensure targets is a list (handle case where it might be a string or other type)
            if not isinstance(targets, list):
                # If targets is not a list, wrap it in a list
                targets = [targets]
            
            # truncate stats to only process one sample at a time
            truncated_stats = {
                k: [v[i]]
                for k, v in stats.items()
                if k in self.stats_dependencies + ["greedy_texts"]
            }

            sample_metric_values = []
            for target in targets:
                # Ensure target is a string (handle nested lists or other types)
                if isinstance(target, list):
                    # If target is a list, take the first element (shouldn't happen, but be defensive)
                    if len(target) > 0:
                        target = target[0] if isinstance(target[0], str) else str(target[0])
                    else:
                        target = ""
                elif not isinstance(target, str):
                    # Convert to string if it's not already
                    target = str(target)
                
                value = self.base_metric(truncated_stats, [target])
                sample_metric_values.append(value)

            if self.aggregation == "max":
                metric_values.append(np.nanmax(sample_metric_values))
            else:
                raise ValueError(f"Unknown aggregation type: {self.aggregation}")

        return np.array(metric_values)
