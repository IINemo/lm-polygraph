from lm_polygraph.utils.deberta import SingletonDeberta
from lm_polygraph.stat_calculators.cross_encoder_similarity import CrossEncoderSimilarityMatrixCalculator


def load_stat_calculator(config, builder):
    nli_model = SingletonDeberta(**config.nli_model)
    return CrossEncoderSimilarityMatrixCalculator(nli_model, config.cross_encoder_name)
