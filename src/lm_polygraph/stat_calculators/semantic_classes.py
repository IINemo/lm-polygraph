import numpy as np

from collections import defaultdict
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class SemanticClassesCalculator(StatCalculator):
    """
    Paritions samples into semantic classes based on semantic matrix.
    """

    def __init__(self):
        super().__init__(
            [
                "semantic_classes_entail",
            ],
            [
                "sample_texts",
                "semantic_matrix_entail",
                "semantic_matrix_classes",
                "entailment_id",
            ],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        self._is_entailment = (
            dependencies["semantic_matrix_classes"] == dependencies["entailment_id"]
        )
        self.get_classes(dependencies["sample_texts"])

        return {
            "semantic_classes_entail": {
                "sample_to_class": self._sample_to_class,
                "class_to_sample": self._class_to_sample,
            }
        }

    def get_classes(self, hyps_list: List[List[str]]):
        self._sample_to_class = {}
        self._class_to_sample: Dict[int, List] = defaultdict(list)

        [
            self._determine_class(idx, i)
            for idx, hyp in enumerate(hyps_list)
            for i in range(len(hyp))
        ]

        return self._sample_to_class, self._class_to_sample

    def _determine_class(self, idx: int, i: int):
        # For first hypo just create a zeroth class
        if i == 0:
            self._class_to_sample[idx] = [[0]]
            self._sample_to_class[idx] = {0: 0}

            return 0

        # Iterate over existing classes and return if hypo belongs to one of them
        for class_id in range(len(self._class_to_sample[idx])):
            class_text_id = self._class_to_sample[idx][class_id][0]
            forward_entailment = self._is_entailment[idx, class_text_id, i]
            backward_entailment = self._is_entailment[idx, i, class_text_id]
            if forward_entailment and backward_entailment:
                self._class_to_sample[idx][class_id].append(i)
                self._sample_to_class[idx][i] = class_id

                return class_id

        # If none of the existing classes satisfy - create new one
        new_class_id = len(self._class_to_sample[idx])
        self._sample_to_class[idx][i] = new_class_id
        self._class_to_sample[idx].append([i])

        return new_class_id
