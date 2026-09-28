import numpy as np
import pytest

from lm_polygraph.normalizers.binned_pcc import BinnedPCCNormalizer


@pytest.fixture
def normalizer():
    normalizer = BinnedPCCNormalizer()
    normalizer.fit(
        gen_metrics=np.array([1.0, 1.0, 0.8, 0.8, 0.2, 0.2]),
        ues=np.arange(6, dtype=float),
        num_bins=3,
    )
    return normalizer


@pytest.mark.parametrize(
    "ue, expected",
    [
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 0.8),
        (3.0, 0.8),
        (4.0, 0.2),
        (4.5, 0.2),
        (5.0, 0.2),
        (6.0, 0.2),
    ],
)
def test_transform_uses_fitted_bins(normalizer, ue, expected):
    np.testing.assert_allclose(normalizer.transform(np.array([ue])), [expected])


def test_transform_boundaries_after_serialization(normalizer):
    restored = BinnedPCCNormalizer.loads(normalizer.dumps())
    np.testing.assert_allclose(
        restored.transform(np.array([0.0, 2.0, 4.0, 5.0])),
        [1.0, 0.8, 0.2, 0.2],
    )


def test_transform_single_bin():
    normalizer = BinnedPCCNormalizer()
    normalizer.fit(
        gen_metrics=np.array([0.2, 0.8]),
        ues=np.array([0.0, 1.0]),
        num_bins=1,
    )
    np.testing.assert_allclose(
        normalizer.transform(np.array([-1.0, 0.0, 0.5, 1.0, 2.0])),
        np.full(5, 0.5),
    )
