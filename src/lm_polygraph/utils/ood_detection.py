import numpy as np


def calculate_ood_from_mans(manager_id, manager_ood, ood_metrics):
    ue_methods_id = set([m[1] for m in manager_id.estimations.keys()])
    ue_methods_ood = set([m[1] for m in manager_ood.estimations.keys()])
    ue_methods = ue_methods_id.intersection(ue_methods_ood)

    results = {}
    for ood_metric in ood_metrics:
        results[str(ood_metric)] = {}
        for ue_method in ue_methods:
            ue_id = manager_id.estimations[("sequence", ue_method)]
            ue_ood = manager_ood.estimations[("sequence", ue_method)]
            ood_labels = [0] * len(ue_id) + [1] * len(ue_ood)
            ue = np.concatenate([ue_id, ue_ood])
            results[str(ood_metric)][ue_method] = ood_metric(ue, ood_labels)
    return results
