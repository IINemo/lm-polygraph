from typing import List, Optional
from omegaconf import OmegaConf

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)


def register_default_stat_calculators(
    model_type: str,
    language: str = "en",
    hf_cache: Optional[str] = None,
    output_attentions: bool = True,
) -> List[StatCalculatorContainer]:
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

    if language == "en":
        deberta_model_path = "microsoft/deberta-large-mnli"
    else:
        deberta_model_path = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    _register(InitialStateCalculator)
    _register(
        SemanticMatrixCalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_SemanticMatrixCalculator",
        {
            "nli_model": {
                "deberta_path": deberta_model_path,
                "batch_size": 10,
                "device": None,
            }
        },
    )
    _register(SemanticClassesCalculator)

    if model_type == "Blackbox":
        _register(BlackboxGreedyTextsCalculator)
        _register(BlackboxSamplingGenerationCalculator)

    elif model_type == "Whitebox":
        _register(
            GreedyProbsCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyProbsCalculator",
            {
                "output_attentions": True,
            },
        )
        _register(EntropyCalculator)
        _register(GreedyLMProbsCalculator)
        _register(PromptCalculator)
        _register(SamplingGenerationCalculator)
        _register(BartScoreCalculator)
        _register(ModelScoreCalculator)
        _register(EnsembleTokenLevelDataCalculator)
        _register(PromptCalculator)
        _register(SamplingPromptCalculator)
        _register(ClaimPromptCalculator)
        _register(
            CrossEncoderSimilarityMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_CrossEncoderSimilarityMatrixCalculator",
            {
                "batch_size": 10,
                "cross_encoder_name": "cross-encoder/stsb-roberta-large",
            },
        )
        _register(
            GreedyAlternativesNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
            {
                "nli_model": {
                    "deberta_path": deberta_model_path,
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
                    "deberta_path": deberta_model_path,
                    "batch_size": 10,
                    "device": None,
                }
            },
        )
        _register(
            ClaimsExtractor,
            "lm_polygraph.defaults.stat_calculator_builders.default_ClaimsExtractor",
            {"openai_model": "gpt-4o", "cache_path": "~/.cache", "language": language},
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return all_stat_calculators