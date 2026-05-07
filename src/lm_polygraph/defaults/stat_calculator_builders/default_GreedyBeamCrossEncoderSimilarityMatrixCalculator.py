from lm_polygraph.stat_calculators.greedy_cross_encoder_similarity import (
    GreedyBeamCrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return GreedyBeamCrossEncoderSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
