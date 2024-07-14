from lm_polygraph.utils.deberta import SingletonDeberta
from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator


def load_stat_calculator(config, builder):
    nli_model = SingletonDeberta(**config.nli_model)
    return SemanticMatrixCalculator(nli_model)
