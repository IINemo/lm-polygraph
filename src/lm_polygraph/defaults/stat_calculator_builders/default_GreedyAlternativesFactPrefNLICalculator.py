from lm_polygraph.utils.deberta import SingletonDeberta
from lm_polygraph.stat_calculators.greedy_alternatives_nli import GreedyAlternativesFactPrefNLICalculator


def load_stat_calculator(config, builder):
    nli_model = SingletonDeberta(**config.nli_model)
    return GreedyAlternativesFactPrefNLICalculator(nli_model)
