from lm_polygraph.stat_calculators.cross_encoder_similarity import (
    CrossEncoderBeamSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return CrossEncoderBeamSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
