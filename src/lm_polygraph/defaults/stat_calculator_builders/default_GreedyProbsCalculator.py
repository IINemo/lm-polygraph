from lm_polygraph.stat_calculators.cross_encoder_similarity import GreedyProbsCalculator


def load_stat_calculator(config, builder):
    return GreedyProbsCalculator(config.output_attentions)
