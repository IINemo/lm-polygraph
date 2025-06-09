import numpy as np


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
