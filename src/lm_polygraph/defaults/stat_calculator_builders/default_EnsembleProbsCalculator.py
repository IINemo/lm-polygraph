from lm_polygraph.stat_calculators.ensemble_probs import (
    EnsembleProbsCalculator,
)


def load_stat_calculator(config, environment):
    return EnsembleProbsCalculator(
        instructions=config["instructions"],
        class_labels=config["class_labels"],
        few_shot_examples=config.get("few_shot_examples", None),
        prompt_formatting=config.get("prompt_formatting", None),
    )
