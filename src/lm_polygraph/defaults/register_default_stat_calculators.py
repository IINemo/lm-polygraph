from typing import Dict, List, Tuple

from lm_polygraph.stat_calculators import *


def register_default_stat_calculators() -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
    """
    Registers all available statistic calculators to be seen by UEManager
    for properly organizing the calculations order.
    """
    stat_calculators: Dict[str, "StatCalculator"] = {}
    stat_dependencies: Dict[str, List[str]] = {}

    def _register(calculator_class: StatCalculator):
        stats, dependencies = calculator_class.meta_info()
        for stat in stats:
            if stat in stat_calculators.keys():
                continue
            stat_calculators[stat] = calculator_class
            stat_dependencies[stat] = dependencies

    _register(GreedyProbsCalculator)
    _register(BlackboxGreedyTextsCalculator)
    _register(EntropyCalculator)
    _register(GreedyLMProbsCalculator)
    _register(PromptCalculator)
    _register(SamplingPromptCalculator)
    _register(ClaimPromptCalculator)
    _register(SamplingGenerationCalculator)
    _register(BlackboxSamplingGenerationCalculator)
    _register(BartScoreCalculator)
    _register(ModelScoreCalculator)
    _register(EmbeddingsCalculator)
    _register(EnsembleTokenLevelDataCalculator)
    _register(SemanticMatrixCalculator)
    _register(CrossEncoderSimilarityMatrixCalculator)
    _register(GreedyProbsCalculator)
    _register(GreedyAlternativesNLICalculator)
    _register(GreedyAlternativesFactPrefNLICalculator)
    _register(ClaimsExtractor)

    return stat_calculators, stat_dependencies