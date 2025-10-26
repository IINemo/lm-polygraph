from lm_polygraph.stat_calculators.greedy_cross_encoder_similarity import (
    GreedyCrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return GreedyCrossEncoderSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
