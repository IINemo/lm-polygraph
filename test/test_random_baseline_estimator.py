import numpy as np

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import RandomBaseline
from lm_polygraph.utils.manager import order_calculators
from lm_polygraph.utils.factory_estimator import FactoryEstimator


def test_random_baseline_scores_shape_and_range():
    estimator = RandomBaseline()
    scores = estimator({"input_texts": ["a", "b", "c"]})

    assert scores.shape == (3,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_random_baseline_has_no_stat_dependencies():
    estimator = RandomBaseline()
    assert estimator.stats_dependencies == []
    assert estimator.level == "sequence"


def test_random_baseline_available_in_factory():
    estimator = FactoryEstimator()("RandomBaseline", {})
    assert isinstance(estimator, RandomBaseline)


def test_random_baseline_resolves_to_greedy_probs_calculator_only():
    estimator = RandomBaseline()
    calculators = register_default_stat_calculators(model_type="Whitebox")

    stat_calculators = {}
    stat_dependencies = {}
    for calculator in calculators:
        for stat in calculator.stats:
            stat_calculators[stat] = calculator
            stat_dependencies[stat] = calculator.dependencies

    ordered_stats, _ = order_calculators(
        estimator.stats_dependencies + ["greedy_texts", "greedy_tokens"],
        stat_calculators,
        stat_dependencies,
    )

    assert ordered_stats == ["greedy_texts"]
    assert stat_calculators[ordered_stats[0]].name == "GreedyProbsCalculator"
