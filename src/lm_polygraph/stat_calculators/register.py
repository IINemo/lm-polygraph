from lm_polygraph.stat_calculators import *

from typing import Dict, List, Optional, Tuple


def register_stat_calculators(
    deberta_batch_size: int = 10,
    deberta_device: Optional[str] = None,
    n_ccp_alternatives: int = 10,
) -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
    """
    Registers all available statistic calculators to be seen by UEManager for properly organizing the calculations
    order.
    """
    stat_calculators: Dict[str, "StatCalculator"] = {}
    stat_dependencies: Dict[str, List[str]] = {}

    def _register(calculator_class: StatCalculator):
        for stat in calculator_class.stats:
            if stat in stat_calculators.keys():
                continue
            stat_calculators[stat] = calculator_class
            stat_dependencies[stat] = calculator_class.stat_dependencies

    _register(GreedyProbsCalculator())
    _register(BlackboxGreedyTextsCalculator())
    _register(EntropyCalculator())
    _register(GreedyLMProbsCalculator())
    _register(
        PromptCalculator(
            "Question: {q}\n Possible answer:{a}\n "
            "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
            "True",
            "p_true",
        )
    )
    _register(
        PromptCalculator(
            "Question: {q}\n Here are some ideas that were brainstormed: {s}\n Possible answer:{a}\n "
            "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
            "True",
            "p_true_sampling",
        )
    )
    _register(SamplingGenerationCalculator())
    _register(BlackboxSamplingGenerationCalculator())
    _register(BartScoreCalculator())
    _register(ModelScoreCalculator())
    _register(EmbeddingsCalculator())
    _register(EnsembleTokenLevelDataCalculator())
    _register(SemanticMatrixCalculator())
    _register(CrossEncoderSimilarityMatrixCalculator())
    _register(Deberta(batch_size=deberta_batch_size, device=deberta_device))
    _register(GreedyProbsCalculator(n_alternatives=n_ccp_alternatives))
    _register(GreedyTokensAlternativesNLICalculator())

    return stat_calculators, stat_dependencies
