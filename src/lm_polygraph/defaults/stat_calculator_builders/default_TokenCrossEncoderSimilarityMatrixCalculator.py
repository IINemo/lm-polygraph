from lm_polygraph.stat_calculators.cross_encoder_similarity import (
    TokenCrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return TokenCrossEncoderSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
