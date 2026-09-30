import numpy as np
import pytest

from lm_polygraph.normalizers.minmax import MinMaxNormalizer
from lm_polygraph.normalizers.quantile import QuantileNormalizer


@pytest.mark.parametrize("normalizer_class", [MinMaxNormalizer, QuantileNormalizer])
def test_normalizers_rank_clip_and_roundtrip(normalizer_class):
    normalizer = normalizer_class()
    normalizer.fit(np.array([0.0, 1.0, 2.0]))
    estimates = np.array([-1.0, 0.5, 1.5, 3.0])
    scores = normalizer.transform(estimates)
    assert scores[0] == pytest.approx(1)
    assert scores[-1] == pytest.approx(0)
    assert np.all(np.diff(scores) <= 0)
    restored = normalizer_class.loads(normalizer.dumps())
    np.testing.assert_allclose(restored.transform(estimates), scores)
