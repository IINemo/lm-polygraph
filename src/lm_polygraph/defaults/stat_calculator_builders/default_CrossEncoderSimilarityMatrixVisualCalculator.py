from lm_polygraph.stat_calculators.cross_encoder_visual_similarity import (
    CrossEncoderSimilarityMatrixVisualCalculator,
)


def load_stat_calculator(config, builder):
    return CrossEncoderSimilarityMatrixVisualCalculator(
        config.batch_size, config.cross_encoder_name
    )
