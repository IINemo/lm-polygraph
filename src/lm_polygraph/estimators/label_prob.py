import numpy as np

from typing import Dict

from .estimator import Estimator


class LabelProb(Estimator):
    def __init__(self):
        super().__init__(["semantic_classes_entail"], "sequence")

    def __str__(self):
        return "LabelProb"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        batch_sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        ues = []
        for batch_i, class_to_sample in batch_class_to_sample.items():
            num_samples = len(batch_sample_to_class[batch_i])
            class_sizes = {k: len(v) for k, v in class_to_sample.items()}
            largest_class = max(class_sizes, key=class_sizes.get)
            ues.append(class_sizes[largest_class] / num_samples)

        return np.array(ues)
