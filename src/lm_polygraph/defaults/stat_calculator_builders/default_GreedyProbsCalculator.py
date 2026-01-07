from lm_polygraph.stat_calculators.greedy_probs import (
    GreedyProbsCalculator,
)


def load_stat_calculator(config, builder):
    n_alternatives = getattr(config, "n_alternatives", 10)
    return GreedyProbsCalculator(
        output_attentions=config.output_attentions,
        output_hidden_states=config.output_hidden_states,
        n_alternatives=n_alternatives,
    )
