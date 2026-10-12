import numpy as np
import pytest

from lm_polygraph.ue_metrics import ECE, PRAUC, ROCAUC, RiskCoverageCurveAUC
from lm_polygraph.ue_metrics.ue_metric import (
    normalize,
    normalize_metric,
    skip_target_nans,
)


@pytest.mark.parametrize(
    "values, expected",
    [([2, 4, 6], [0, 0.5, 1]), ([7, 7], [0.5, 0.5])],
)
def test_normalize(values, expected):
    np.testing.assert_allclose(normalize(values), expected)


def test_missing_labels_preserve_alignment():
    target, estimates = skip_target_nans([1, np.nan, 0], [0.1, 99, 0.9])
    assert target == [1, 0]
    assert estimates == [0.1, 0.9]


def test_metric_normalization():
    assert normalize_metric(0.75, 1, 0.5) == pytest.approx(0.5)
    assert normalize_metric(0.75, 1, 1) == pytest.approx(0.75)


@pytest.mark.parametrize("metric", [ROCAUC(), PRAUC()])
def test_auc_ignores_unlabeled_examples(metric):
    assert metric([0.1, 99, 0.9], [0, np.nan, 1]) == pytest.approx(1)


def test_risk_coverage_prefers_correct_ranking():
    metric = RiskCoverageCurveAUC()
    assert metric([0, 1], [1, 0]) == pytest.approx(0.25)
    assert metric([1, 0], [1, 0]) == pytest.approx(0.75)


def test_perfect_calibration_includes_bin_boundaries():
    assert ECE(n_bins=2)([-1, 0], [1, 0]) == pytest.approx(0)


def test_calibration_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        ECE()([-1], [1, 0])
