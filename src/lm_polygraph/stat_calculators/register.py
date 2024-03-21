from lm_polygraph.stat_calculators import *

from typing import Dict, List, Optional

STAT_CALCULATORS: Dict[str, "StatCalculator"] = {}
STAT_DEPENDENCIES: Dict[str, List[str]] = {}


def _register(calculator_class: StatCalculator):
    """
    Registers a new statistics calculator to be seen by UEManager for properly organizing the calculations order.
    Needs to be called at lm_polygraph/stat_calculators/__init__.py for all stat calculators used in running benchmarks.
    """
    for stat in calculator_class.stats:
        if stat in STAT_CALCULATORS.keys():
            continue
        STAT_CALCULATORS[stat] = calculator_class
        STAT_DEPENDENCIES[stat] = calculator_class.stat_dependencies


def register_stat_calculators(
    deberta_batch_size: int = 10,
    deberta_device: Optional[str] = None,
    n_ccp_alternatives: int = 10,
):
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
    _register(GreedyAlternativesNLICalculator())
