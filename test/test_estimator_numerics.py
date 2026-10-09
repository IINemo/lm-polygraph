"""Numerical estimator contracts using small, model-independent statistics."""

import numpy as np
import pytest

from lm_polygraph.estimators import (
    ConditionalPointwiseMutualInformation,
    MaximumSequenceProbability,
    MaximumTokenProbability,
    MeanConditionalPointwiseMutualInformation,
    MeanPointwiseMutualInformation,
    MeanTokenEntropy,
    MonteCarloNormalizedSequenceEntropy,
    MonteCarloSequenceEntropy,
    Perplexity,
    PointwiseMutualInformation,
    TokenEntropy,
)
from lm_polygraph.stat_calculators import EntropyCalculator


@pytest.fixture
def generation_stats():
    # Ragged batches; the last entry in each generation represents EOS.
    # The large first unconditional score detects accidental PMI correction
    # at position zero. Entropies straddle and equal the CPMI threshold.
    return {
        "greedy_log_likelihoods": [[-1.0, -2.0, -3.0], [-4.0, -2.0]],
        "greedy_lm_log_likelihoods": [[-9.0, -1.0, -2.0], [-8.0, -0.5]],
        "entropy": [[0.25, 0.5, 1.0], [0.75, 0.25]],
    }


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (MaximumSequenceProbability(), [6.0, 6.0]),
        # Perplexity returns negative mean log likelihood, without exponentiation.
        (Perplexity(), [2.0, 3.0]),
        (MeanTokenEntropy(), [7.0 / 12.0, 0.5]),
        (MeanPointwiseMutualInformation(), [1.0, 2.75]),
        (
            MeanConditionalPointwiseMutualInformation(tau=0.5, lambd=2.0),
            [0.0, 3.0],
        ),
    ],
    ids=lambda value: str(value),
)
def test_sequence_scores(generation_stats, estimator, expected):
    scores = estimator(generation_stats)

    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (MaximumTokenProbability(), [[1.0, 2.0], [4.0]]),
        (TokenEntropy(), [[0.25, 0.5], [0.75]]),
    ],
    ids=lambda value: str(value),
)
def test_ragged_token_scores_exclude_eos(generation_stats, estimator, expected):
    scores = estimator(generation_stats)

    assert len(scores) == 2
    assert [score.shape for score in scores] == [(2,), (1,)]
    for score, expected_score in zip(scores, expected):
        np.testing.assert_allclose(score, expected_score)


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (PointwiseMutualInformation(), [1.0, 1.0, 4.0]),
        (ConditionalPointwiseMutualInformation(tau=0.5, lambd=2.0), [1.0, 0.0, 4.0]),
    ],
    ids=lambda value: str(value),
)
def test_flat_token_scores_exclude_eos(generation_stats, estimator, expected):
    scores = estimator(generation_stats)

    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, expected)


@pytest.fixture
def sampled_stats():
    return {
        "sample_log_probs": [[-2.0, -6.0, -100.0], [-3.0, -9.0]],
        "sample_tokens": [[[1, 2], [3, 4, 5], []], [[1], [2, 3, 4]]],
    }


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (MonteCarloSequenceEntropy(), [36.0, 6.0]),
        # The empty sampled generation is excluded from the normalized mean.
        (MonteCarloNormalizedSequenceEntropy(), [1.5, 3.0]),
    ],
    ids=lambda value: str(value),
)
def test_sampled_entropy_scores(sampled_stats, estimator, expected):
    scores = estimator(sampled_stats)

    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, expected)


def test_entropy_calculator_known_distributions():
    stats = {
        "greedy_log_probs": [
            np.array([[0.0, -np.inf], [-np.log(2.0), -np.log(2.0)]]),
            np.log(np.array([[0.25, 0.75]])),
            np.empty((0, 2)),
        ]
    }

    entropies = EntropyCalculator()(stats)["entropy"]

    assert [len(entropy) for entropy in entropies] == [2, 1, 0]
    np.testing.assert_allclose(entropies[0], [0.0, np.log(2.0)])
    np.testing.assert_allclose(
        entropies[1], [-0.25 * np.log(0.25) - 0.75 * np.log(0.75)]
    )
    assert np.isfinite(entropies[0]).all()


@pytest.mark.parametrize(
    "estimator",
    [
        Perplexity(),
        MeanTokenEntropy(),
        MeanPointwiseMutualInformation(),
        MeanConditionalPointwiseMutualInformation(),
        MonteCarloSequenceEntropy(),
        MonteCarloNormalizedSequenceEntropy(),
    ],
    ids=str,
)
def test_empty_generation_average_is_undefined(estimator):
    stats = {dependency: [[]] for dependency in estimator.stats_dependencies}

    # Current contract: averaging no observations yields NaN, not confidence 0.
    with pytest.warns(RuntimeWarning):
        scores = estimator(stats)

    assert scores.shape == (1,)
    assert np.isnan(scores[0])


def test_empty_generation_sequence_log_probability():
    scores = MaximumSequenceProbability()({"greedy_log_likelihoods": [[]]})

    assert scores.shape == (1,)
    np.testing.assert_array_equal(scores, [0.0])


@pytest.mark.parametrize(
    "estimator",
    [
        MaximumTokenProbability(),
        TokenEntropy(),
        PointwiseMutualInformation(),
        ConditionalPointwiseMutualInformation(),
    ],
    ids=str,
)
@pytest.mark.parametrize("generation", [[], [-1.0]], ids=["empty", "eos-only"])
def test_no_content_tokens_have_empty_scores(estimator, generation):
    stats = {dependency: [generation] for dependency in estimator.stats_dependencies}

    scores = estimator(stats)

    if isinstance(scores, list):
        assert len(scores) == 1
        scores = scores[0]
    assert scores.shape == (0,)
