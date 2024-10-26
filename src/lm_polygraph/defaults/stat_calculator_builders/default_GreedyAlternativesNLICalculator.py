from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.stat_calculators.greedy_alternatives_nli import (
    GreedyAlternativesNLICalculator,
)


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = Deberta(**config.nli_model)

    return GreedyAlternativesNLICalculator(builder.nli_model)
