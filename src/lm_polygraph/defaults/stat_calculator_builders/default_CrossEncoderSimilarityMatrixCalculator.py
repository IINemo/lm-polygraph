from lm_polygraph.stat_calculators.cross_encoder_similarity import (
    CrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return CrossEncoderSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
