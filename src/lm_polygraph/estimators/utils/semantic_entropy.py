import numpy as np
from typing import List, Optional

def compute_semantic_entropy(
    hyps_list: List[List[str]],
    loglikelihoods_list: Optional[List[List[float]]],
    class_to_sample: List[List[int]],
    sample_to_class: List[List[int]],
    class_probability_estimation: str = "sum",
    entropy_estimator: str = "mean",
) -> np.ndarray:
    """
    Core logic to compute sequence-level semantic entropy.

    Args:
        hyps_list: List of generation samples per input.
        loglikelihoods_list: List of log-probabilities per sample, or None if using frequency.
        class_to_sample: Mapping from class indices to sample indices.
        sample_to_class: Mapping from sample indices to class indices.
        class_probability_estimation: "sum" for sum over log-probs, "frequency" for empirical frequency.
        entropy_estimator: "mean" for mean entropy, "direct" for sum of p*log p entropy.

    Returns:
        np.ndarray of semantic entropy values per input.
    """
    results = []
    for i, hyps in enumerate(hyps_list):
        if class_probability_estimation == "sum":
            likelihoods = loglikelihoods_list[i]
            class_likelihoods = [
                np.array(likelihoods)[np.array(cls)]
                for cls in class_to_sample[i]
            ]
            class_lp = [np.logaddexp.reduce(likelihood) for likelihood in class_likelihoods]
        elif class_probability_estimation == "frequency":
            num = len(hyps)
            class_lp = np.log([len(cls) / num for cls in class_to_sample[i]])
        else:
            raise ValueError(f"Unknown class_probability_estimation: {class_probability_estimation}")

        if entropy_estimator == "mean":
            ent = -np.mean(
                [class_lp[sample_to_class[i][j]] for j in range(len(hyps))]
            )
        elif entropy_estimator == "direct":
            ent = -np.sum(
                [
                    class_lp[sample_to_class[i][j]]
                    * np.exp(class_lp[sample_to_class[i][j]])
                    for j in range(len(hyps))
                ]
            )
        else:
            raise ValueError(f"Unknown entropy_estimator: {entropy_estimator}")

        results.append(ent)

    return np.array(results)
