from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator


def load_stat_calculator(config, builder):
    return GreedyProbsCalculator(config.output_attentions, config.output_hidden_states)
