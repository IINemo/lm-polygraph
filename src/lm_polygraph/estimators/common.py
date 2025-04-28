import numpy as np

SAMPLE_SELECTION_STAT_KEYS = ["best_sample_text_ids", "best_normalized_sample_text_ids"]

def _get_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pairs.append((lst[i], lst[j], i, j))
    return pairs


def _compute_Jaccard_score(lst):
    jaccard_sim_mat = np.eye(len(lst))
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            set1 = set(lst[i].lower().split())
            set2 = set(lst[j].lower().split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            if union == 0:
                jaccard_score = 0
            else:
                jaccard_score = intersection / union
            jaccard_sim_mat[i, j] = jaccard_score
            jaccard_sim_mat[j, i] = jaccard_score

    return jaccard_sim_mat


def compute_sim_score(answers, affinity, similarity_score):
    return _compute_Jaccard_score(answers)


def sample_strategy_to_prefix(sample_strategy):
    if sample_strategy == "first":
        return ""
    elif sample_strategy in ["best", "best_normalized", "mbr"]:
        return "".join(list(map(lambda x: x.capitalize(), sample_strategy.split("_"))))
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")


def best_sample_ids(sample_strategy, stats):
    batch_size = len(stats["sample_log_probs"])
    if sample_strategy == "first":
        return [0] * batch_size
    elif sample_strategy in ["best", "best_normalized", "mbr"]:
        return stats[f"{sample_strategy}_sample_text_ids"]
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")
