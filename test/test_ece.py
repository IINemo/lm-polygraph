import numpy as np
import pytest

from lm_polygraph.ue_metrics.ece import ECE


@pytest.mark.parametrize("estimator", [[0.2, 0.8], [-0.8, 0.2], [-1.2, -0.8]])
def test_ece_rejects_out_of_range_confidences(estimator):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ECE()(estimator, [0, 1])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_ece_rejects_nonfinite_estimates(normalize, invalid):
    with pytest.raises(ValueError, match="finite"):
        ECE(normalize=normalize)([-0.8, invalid], [0, 1])


@pytest.mark.parametrize("normalize", [False, True])
def test_ece_rejects_empty_inputs(normalize):
    with pytest.raises(ValueError, match="empty"):
        ECE(normalize=normalize)([], [])


def test_ece_counts_confidence_endpoints_and_bin_boundary_once():
    # Confidences 0 and 0.5 share the first bin, and 1 is in the second.
    # Their contributions are (2/3)*abs(0.5-0.25) + (1/3)*abs(0-1).
    assert ECE(n_bins=2)([0.0, -0.5, -1.0], [0, 1, 0]) == pytest.approx(0.5)


def test_ece_perfect_confidences():
    assert ECE()([0.0, -1.0], [0, 1]) == pytest.approx(0.0)


def test_ece_explicit_normalization_accepts_unbounded_finite_estimates():
    # These estimates become confidences [1, 0.5, 0].
    assert ECE(normalize=True, n_bins=2)([10, 20, 30], [1, 0, 0]) == pytest.approx(
        1 / 6
    )
