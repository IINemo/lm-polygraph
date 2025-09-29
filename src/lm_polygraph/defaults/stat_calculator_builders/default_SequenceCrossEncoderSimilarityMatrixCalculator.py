from lm_polygraph.stat_calculators.cross_encoder_similarity import (
    SequenceCrossEncoderSimilarityMatrixCalculator,
)


def load_stat_calculator(config, builder):
    return SequenceCrossEncoderSimilarityMatrixCalculator(
        config.batch_size, config.cross_encoder_name
    )
