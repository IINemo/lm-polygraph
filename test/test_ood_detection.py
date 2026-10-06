import numpy as np
from types import SimpleNamespace

from lm_polygraph.ue_metrics import ROCAUC
from lm_polygraph.utils.ood_detection import calculate_ood_from_mans


def _man(**methods):
    return SimpleNamespace(
        estimations={("sequence", name): vals for name, vals in methods.items()}
    )


def test_ood_roc_auc_separable():
    man_id = _man(m=[0.1, 0.2, 0.3])
    man_ood = _man(m=[0.7, 0.8, 0.9])
    res = calculate_ood_from_mans(man_id, man_ood, [ROCAUC()])
    assert res["roc-auc"]["m"] == 1.0


def test_ood_skips_nan_and_inf():
    man_id = _man(m=[0.1, np.nan, 0.3], only_id=[0.1])
    man_ood = _man(m=[0.7, 0.8, np.inf], only_ood=[0.1])
    res = calculate_ood_from_mans(man_id, man_ood, [ROCAUC()])
    assert list(res["roc-auc"]) == ["m"]  # only common methods
    assert res["roc-auc"]["m"] == 1.0
