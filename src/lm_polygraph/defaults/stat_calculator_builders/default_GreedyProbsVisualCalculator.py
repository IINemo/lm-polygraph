from lm_polygraph.stat_calculators.greedy_visual_probs import (
    GreedyProbsVisualCalculator,
)


def load_stat_calculator(config, builder):
    return GreedyProbsVisualCalculator(config.output_attentions)
