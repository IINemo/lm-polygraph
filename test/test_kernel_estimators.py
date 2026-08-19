import numpy as np
import pytest
import scipy

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators.predictive_kernel_entropy import (
    PredictiveKernelEntropy,
)
from lm_polygraph.estimators.spectral_uncertainty import SpectralUncertainty


def test_predictive_kernel_entropy_cosine_normalizes_embeddings():
    estimator = PredictiveKernelEntropy(kernel="cosine")
    stats = {"sample_sentence_embeddings": [[[1.0, 0.0], [2.0, 0.0]]]}

    assert estimator(stats) == pytest.approx([-1.0])


def test_spectral_uncertainty_cosine_is_scale_invariant():
    estimator = SpectralUncertainty(kernel="cosine")
    stats = {"sample_sentence_embeddings": [[[2.0, 0.0], [0.0, 1.0]]]}

    assert estimator(stats) == pytest.approx([np.log(2.0)])


def test_spectral_uncertainty_uses_configured_eps(monkeypatch):
    estimator = SpectralUncertainty(eps=0.25)
    estimator.kernel_function = lambda embeddings: np.eye(len(embeddings))
    stats = {"sample_sentence_embeddings": [[[1.0], [2.0]]]}
    original_scipy_eigh = scipy.linalg.eigh
    matrices = []

    def scipy_eigh(matrix):
        if not matrices:
            matrices.append(matrix.copy())
            raise np.linalg.LinAlgError
        matrices.append(matrix.copy())
        return original_scipy_eigh(matrix)

    def numpy_eigh(matrix):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(scipy.linalg, "eigh", scipy_eigh)
    monkeypatch.setattr(np.linalg, "eigh", numpy_eigh)

    estimator(stats)

    np.testing.assert_allclose(matrices[1], matrices[0] + 0.25 * np.eye(2))
    assert "eps=0.25" in str(estimator)


def test_sentence_embedder_uses_hf_cache(tmp_path):
    hf_cache = str(tmp_path / "huggingface")

    calculators = register_default_stat_calculators("Whitebox", hf_cache=hf_cache)
    sentence_embeddings = next(
        calculator
        for calculator in calculators
        if calculator.name == "SampleSentenceEmbeddingsCalculator"
    )

    assert sentence_embeddings.cfg.sentence_embedding_model.cache_folder == hf_cache
