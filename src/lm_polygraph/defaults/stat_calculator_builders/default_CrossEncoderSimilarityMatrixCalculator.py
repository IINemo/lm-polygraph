from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.stat_calculators.cross_encoder_similarity import (
    CrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    if not hasattr(builder, "nli_model"):
        builder.nli_model = Deberta(**config.nli_model)

    return CrossEncoderSimilarityMatrixCalculator(builder.nli_model, config.cross_encoder_name)
