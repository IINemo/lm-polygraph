import numpy as np
import pytest

from lm_polygraph.estimators.semantic_entropy import SemanticEntropy


def make_stats(texts, classes, log_probs=None):
    stats = {
        "sample_texts": [texts],
        "semantic_classes_entail": {
            "class_to_sample": {0: classes},
            "sample_to_class": {
                0: {
                    sample_id: class_id
                    for class_id, samples in enumerate(classes)
                    for sample_id in samples
                }
            },
        },
    }
    if log_probs is not None:
        stats["sample_log_probs"] = [log_probs]
    return stats


@pytest.mark.parametrize("entropy_estimation", ["direct", "mean"])
@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_empirical_entropy_is_invariant_to_repeating_samples(
    entropy_estimation, repeats
):
    texts = ["Paris"] * (2 * repeats) + ["London"] * repeats
    classes = [list(range(2 * repeats)), list(range(2 * repeats, 3 * repeats))]
    estimator = SemanticEntropy(
        class_probability_estimation="frequency",
        entropy_estimation=entropy_estimation,
    )

    probabilities = np.array([2 / 3, 1 / 3])
    expected = -np.sum(probabilities * np.log(probabilities))
    np.testing.assert_allclose(estimator(make_stats(texts, classes)), [expected])


def test_direct_empirical_entropy_balanced_classes():
    estimator = SemanticEntropy(
        class_probability_estimation="frequency", entropy_estimation="direct"
    )
    stats = make_stats(["Paris", "Paris", "London", "London"], [[0, 1], [2, 3]])

    np.testing.assert_allclose(estimator(stats), [np.log(2)])


@pytest.mark.parametrize("entropy_estimation", ["direct", "mean"])
def test_empirical_entropy_single_class(entropy_estimation):
    estimator = SemanticEntropy(
        class_probability_estimation="frequency",
        entropy_estimation=entropy_estimation,
    )
    stats = make_stats(["Paris", "Paris", "Paris"], [[0, 1, 2]])

    np.testing.assert_allclose(estimator(stats), [0.0])


@pytest.mark.parametrize("use_unique_responses", [False, True])
def test_direct_entropy_sums_each_class_once(use_unique_responses):
    if use_unique_responses:
        texts = ["Paris", "Paris", "London"]
        log_probs = np.log([0.3, 0.3, 0.7])
    else:
        texts = ["Paris", "It is Paris", "London"]
        log_probs = np.log([0.1, 0.2, 0.7])
    estimator = SemanticEntropy(
        entropy_estimation="direct", use_unique_responses=use_unique_responses
    )
    stats = make_stats(texts, [[0, 1], [2]], log_probs)

    probabilities = np.array([0.3, 0.7])
    expected = -np.sum(probabilities * np.log(probabilities))
    np.testing.assert_allclose(estimator(stats), [expected])


def test_mean_entropy_keeps_sample_weighting():
    estimator = SemanticEntropy()
    stats = make_stats(
        ["Paris", "It is Paris", "London"],
        [[0, 1], [2]],
        np.log([0.1, 0.2, 0.7]),
    )

    expected = -(2 * np.log(0.3) + np.log(0.7)) / 3
    np.testing.assert_allclose(estimator(stats), [expected])


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-12])
def test_direct_entropy_normalizes_class_probabilities(scale):
    estimator = SemanticEntropy(entropy_estimation="direct")
    stats = make_stats(
        ["Paris", "It is Paris", "London"],
        [[0, 1], [2]],
        np.log(scale * np.array([0.1, 0.2, 0.1])),
    )

    probabilities = np.array([0.75, 0.25])
    expected = -np.sum(probabilities * np.log(probabilities))
    np.testing.assert_allclose(estimator(stats), [expected])
