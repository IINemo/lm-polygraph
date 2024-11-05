from typing import List
from omegaconf import OmegaConf

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)


def register_default_stat_calculators(model_type: str) -> List[StatCalculatorContainer]:
    """
    Specifies the list of the default stat_calculators that could be used in the evaluation scripts and
    estimate_uncertainty() function with default configurations.
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

    if model_type == "Blackbox":
        _register(BlackboxGreedyTextsCalculator)
        _register(BlackboxSamplingGenerationCalculator)

    elif model_type == "Whitebox":
        _register(GreedyProbsCalculator)
        _register(EntropyCalculator)
        _register(GreedyLMProbsCalculator)
        _register(PromptCalculator)
        _register(SamplingGenerationCalculator)
        _register(BartScoreCalculator)
        _register(ModelScoreCalculator)
        _register(EnsembleTokenLevelDataCalculator)
        _register(SemanticClassesCalculator)
        _register(PromptCalculator)
        _register(SamplingPromptCalculator)
        _register(ClaimPromptCalculator)
        _register(
            CrossEncoderSimilarityMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_CrossEncoderSimilarityMatrixCalculator",
            {
                "nli_model": {
                    "deberta_path": "microsoft/deberta-large-mnli",
                    "batch_size": 10,
                    "device": None,
                },
                "cross_encoder_name": "cross-encoder/stsb-roberta-large",
            },
        )
        _register(
            SemanticMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_SemanticMatrixCalculator",
            {
                "nli_model": {
                    "deberta_path": "microsoft/deberta-large-mnli",
                    "batch_size": 10,
                    "device": None,
                }
            },
        )
        _register(
            GreedyAlternativesNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
            {
                "nli_model": {
                    "deberta_path": "microsoft/deberta-large-mnli",
                    "batch_size": 10,
                    "device": None,
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
                    "device": None,
                }
            },
        )
        _register(
            ClaimsExtractor,
            "lm_polygraph.defaults.stat_calculator_builders.default_ClaimsExtractor",
            {"openai_model": "gpt-4o", "cache_path": "~/.cache"},
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return all_stat_calculators
