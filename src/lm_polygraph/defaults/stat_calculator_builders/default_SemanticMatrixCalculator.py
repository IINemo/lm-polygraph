from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = Deberta(**config.nli_model)

    return SemanticMatrixCalculator(builder.nli_model)
