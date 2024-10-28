from typing import Dict, List, Tuple
from omegaconf import OmegaConf

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)


def register_default_stat_calculators() -> List[StatCalculatorContainer]:
    """
    Registers all available statistic calculators to be seen by UEManager
    for properly organizing the calculations order.
    """
    all_stat_calculators = []

    def _register(
        calculator_class: StatCalculator,
        builder="lm_polygraph.utils.builder_stat_calculator_simple",
        default_config=dict(),
    ):
        cfg = dict()
        cfg.update(default_config)
        cfg["obj"] = calculator_class.__name__

        sc = StatCalculatorContainer(
            name=calculator_class.__name__,
            obj=calculator_class,
            builder=builder,
            cfg=OmegaConf.create(cfg),
            dependencies=calculator_class.meta_info()[1],
            stats=calculator_class.meta_info()[0],
        )
        all_stat_calculators.append(sc)
        # stats, dependencies = calculator_class.meta_info()
        # for stat in stats:
        #     if stat in stat_calculators.keys():
        #         continue

        #     dct = dict()
        #     dct.update(default_config)
        #     dct.update({"obj": calculator_class.__name__})
        #     dct = OmegaConf.create(dct)
        #     stat_calculators[stat] = StatCalculatorContainer(
        #         obj=calculator_class,
        #         builder=builder,
        #         cfg=dct,
        #     )
        #     stat_dependencies[stat] = dependencies

    _register(GreedyProbsCalculator)
    _register(BlackboxGreedyTextsCalculator)
    _register(EntropyCalculator)
    _register(GreedyLMProbsCalculator)
    _register(PromptCalculator)
    _register(SamplingGenerationCalculator)
    _register(BlackboxSamplingGenerationCalculator)
    _register(BartScoreCalculator)
    _register(ModelScoreCalculator)
    # _register(EmbeddingsCalculator)
    # _register(EmbeddingsExtractionCalculator)
    _register(EnsembleTokenLevelDataCalculator)
    _register(SemanticMatrixCalculator)
    _register(CrossEncoderSimilarityMatrixCalculator)
    _register(GreedyProbsCalculator)
    # _register(SemanticClassesCalculator)
    _register(
        GreedyAlternativesNLICalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
        {
            "nli_model": {
                "deberta_path": "microsoft/deberta-large-mnli",
                "batch_size": 10,
                "device": "cuda",
            }
        },
    )
    _register(
        GreedyAlternativesFactPrefNLICalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesFactPrefNLICalculator",
        {
            "nli_model": {
                "deberta_path": "microsoft/deberta-large-mnli",
                "batch_size": 10,
                "device": "cuda",
            }
        },
    )
    _register(ClaimsExtractor)

    return all_stat_calculators
    # return stat_calculators, stat_dependencies