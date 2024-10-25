from lm_polygraph.utils.deberta import SingletonDeberta
from lm_polygraph.stat_calculators.greedy_alternatives_nli import (
    GreedyAlternativesNLICalculator,
)


def load_stat_calculator(config, builder):
    nli_model = SingletonDeberta(**config.nli_model)
    return GreedyAlternativesNLICalculator(nli_model)
