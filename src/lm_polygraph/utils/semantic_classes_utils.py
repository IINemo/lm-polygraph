import numpy as np
from typing import List, Dict, Tuple


def calculate_semantic_classes(
    hyps_list: List[List[str]],
    semantic_matrix_classes: np.ndarray,
    entailment_id: int,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, List[List[int]]]]:
    """
    Partition generated hypotheses into semantic classes based on NLI entailment matrix.

    Args:
        hyps_list: batch of lists of generated samples per input.
        semantic_matrix_classes: array of shape [batch, n, n] with NLI class IDs for each pair.
        entailment_id: NLI class ID corresponding to entailment.

    Returns:
        sample_to_class: mapping from input index to a dict of sample index -> class ID.
        class_to_sample: mapping from input index to list of classes, each a list of sample indices.
    """
    is_entailment = semantic_matrix_classes == entailment_id
    sample_to_class: Dict[int, Dict[int, int]] = {}
    class_to_sample: Dict[int, List[List[int]]] = {}

    for idx, hyps in enumerate(hyps_list):
        sample_to_class[idx] = {}
        class_to_sample[idx] = []
        for i in range(len(hyps)):
            if i == 0:
                class_to_sample[idx].append([0])
                sample_to_class[idx][0] = 0
                continue

            # try to assign to existing class
            assigned = False
            for class_id, members in enumerate(class_to_sample[idx]):
                class_text_id = members[0]
                if is_entailment[idx, class_text_id, i] and is_entailment[idx, i, class_text_id]:
                    class_to_sample[idx][class_id].append(i)
                    sample_to_class[idx][i] = class_id
                    assigned = True
                    break
            if not assigned:
                new_class_id = len(class_to_sample[idx])
                class_to_sample[idx].append([i])
                sample_to_class[idx][i] = new_class_id

    return sample_to_class, class_to_sample