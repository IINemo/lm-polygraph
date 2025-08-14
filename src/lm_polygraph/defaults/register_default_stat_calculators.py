from typing import List, Optional
from omegaconf import OmegaConf
from pathlib import Path

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)


def register_default_stat_calculators(
    model_type: str,
    language: str = "en",
    blackbox_supports_logprobs: bool = False,
    output_attentions: bool = True,
    output_hidden_states: bool = True,
    *,
    nli_deberta_path: Optional[str] = None,
    nli_batch_size: Optional[int] = None,
    cross_encoder_name: Optional[str] = None,
    cross_encoder_batch_size: Optional[int] = None,
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

    # Shared NLI model config
    nli_model_cfg = {
        "deberta_path": nli_deberta_path,
        "batch_size": nli_batch_size,
        "device": None,
    }

    _register(InitialStateCalculator)
    _register(RawInputCalculator)
    _register(
        SemanticMatrixCalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_SemanticMatrixCalculator",
        {"nli_model": nli_model_cfg},
    )
    _register(
        GreedySemanticMatrixCalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_GreedySemanticMatrixCalculator",
        {"nli_model": nli_model_cfg},
    )
    _register(
        ConcatGreedySemanticMatrixCalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_ConcatGreedySemanticMatrixCalculator",
        {"nli_model": nli_model_cfg},
    )
    _register(SemanticClassesCalculator)

    if model_type == "Blackbox":
        _register(BlackboxGreedyTextsCalculator)
        _register(BlackboxSamplingGenerationCalculator)
        if blackbox_supports_logprobs:
            # For blackbox models that support logprobs (like OpenAI models)
            _register(EntropyCalculator)
            _register(
                GreedyAlternativesNLICalculator,
                "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
                {"nli_model": nli_model_cfg},
            )

    elif model_type == "Whitebox":
        _register(
            GreedyProbsCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyProbsCalculator",
            {
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
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
        _register(AttentionElicitingPromptCalculator)
        _register(
            CrossEncoderSimilarityMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_CrossEncoderSimilarityMatrixCalculator",
            {
                "batch_size": cross_encoder_batch_size,
                "cross_encoder_name": cross_encoder_name,
            },
        )
        _register(
            GreedyCrossEncoderSimilarityMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyCrossEncoderSimilarityMatrixCalculator",
            {
                "batch_size": cross_encoder_batch_size,
                "cross_encoder_name": cross_encoder_name,
            },
        )
        _register(
            GreedyAlternativesNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
            {"nli_model": nli_model_cfg},
        )
        _register(
            GreedyAlternativesFactPrefNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesFactPrefNLICalculator",
            {"nli_model": nli_model_cfg},
        )
        _register(
            SemanticClassesClaimToSamplesCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_SemanticClassesClaimToSamplesCalculator",
            {"nli_model": nli_model_cfg},
        )
        _register(
            ClaimsExtractor,
            "lm_polygraph.defaults.stat_calculator_builders.default_ClaimsExtractor",
            {
                "openai_model": "gpt-4o",
                "cache_path": str(Path.home() / ".cache"),
                "language": language,
            },
        )
        _register(AttentionForwardPassCalculator)
    elif model_type == "VisualLM":
        _register(
            GreedyProbsVisualCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyProbsVisualCalculator",
            {
                "output_attentions": True,
            },
        )
        _register(EntropyCalculator)
        _register(GreedyLMProbsVisualCalculator)
        _register(PromptVisualCalculator)
        _register(SamplingGenerationVisualCalculator)
        _register(BartScoreCalculator)
        _register(ModelScoreCalculator)
        _register(EnsembleTokenLevelDataCalculator)
        _register(PromptVisualCalculator)
        _register(SamplingPromptVisualCalculator)
        _register(ClaimPromptVisualCalculator)
        _register(
            CrossEncoderSimilarityMatrixVisualCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_CrossEncoderSimilarityMatrixVisualCalculator",
            {
                "batch_size": cross_encoder_batch_size,
                "cross_encoder_name": cross_encoder_name,
            },
        )
        _register(
            GreedyAlternativesNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
            {"nli_model": nli_model_cfg},
        )
        _register(
            GreedyAlternativesFactPrefNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesFactPrefNLICalculator",
            {"nli_model": nli_model_cfg},
        )
        _register(
            ClaimsExtractor,
            "lm_polygraph.defaults.stat_calculator_builders.default_ClaimsExtractor",
            {
                "openai_model": "gpt-4o",
                "cache_path": str(Path.home() / ".cache"),
                "language": language,
            },
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return all_stat_calculators
