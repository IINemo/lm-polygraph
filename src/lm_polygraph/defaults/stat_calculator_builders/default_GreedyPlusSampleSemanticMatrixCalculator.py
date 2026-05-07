from lm_polygraph.stat_calculators.semantic_matrix import GreedyPlusSampleSemanticMatrixCalculator
from .utils import load_nli_model


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = load_nli_model(**config.nli_model)

    return GreedyPlusSampleSemanticMatrixCalculator(builder.nli_model)
