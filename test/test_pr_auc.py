import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from lm_polygraph.ue_metrics.pr_auc import PRAUC


@pytest.mark.parametrize("container", [list, np.array])
@pytest.mark.parametrize(
    "scores, target, expected",
    [
        ([0.0, np.inf], [0, 1], 1.0),
        ([-np.inf, 0.0], [0, 1], 1.0),
        ([-np.inf, np.inf], [0, 1], 1.0),
        ([np.inf, -np.inf], [0, 1], 0.5),
        ([np.inf, np.inf], [0, 1], 0.5),
        ([-np.inf, -np.inf], [0, 1], 0.5),
        ([np.finfo(float).max, np.inf], [0, 1], 1.0),
        ([-np.inf, -np.finfo(float).max], [0, 1], 1.0),
    ],
)
def test_pr_auc_preserves_infinite_score_order_and_ties(
    container, scores, target, expected
):
    assert PRAUC()(container(scores), target) == pytest.approx(expected)


def test_pr_auc_preserves_finite_ties_among_infinities():
    scores = [-np.inf, 0.5, 0.5, 1.0, np.inf, np.inf]
    target = [0, 1, 0, 1, 0, 1]
    expected = average_precision_score(target, [0, 1, 1, 2, 3, 3])
    assert PRAUC()(scores, target) == pytest.approx(expected)


def test_pr_auc_preserves_adjacent_large_finite_scores():
    maximum = np.finfo(float).max
    scores = [np.nextafter(maximum, 0), maximum, np.inf]
    assert PRAUC()(scores, [0, 1, 1]) == pytest.approx(1.0)


def test_pr_auc_preserves_reversed_binary_classes():
    assert PRAUC(positive_class=0, negative_class=1)(
        [-np.inf, np.inf], [0, 1]
    ) == pytest.approx(1.0)


def test_pr_auc_skips_unlabeled_targets():
    assert PRAUC()([np.nan, -np.inf, np.inf], [np.nan, 0, 1]) == pytest.approx(1.0)


def test_pr_auc_does_not_turn_nan_scores_into_valid_ranks():
    with pytest.raises(ValueError):
        PRAUC()([np.nan, np.inf], [0, 1])


def test_pr_auc_finite_scores_match_sklearn():
    scores = [0.1, 0.3, 0.3, 0.8]
    target = [0, 1, 0, 1]
    assert PRAUC()(scores, target) == pytest.approx(
        average_precision_score(target, scores)
    )
