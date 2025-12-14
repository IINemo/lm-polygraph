from typing import Optional, List, Union
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

from lm_polygraph.utils.model import Model, WhiteboxModel
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)

@dataclass
class UncertaintyOutput:
    uncertainty: Union[float, List[float]]
    input_text: str
    generation_text: str
    generation_tokens: List[int]
    model_path: str
    estimator: str

def estimate_uncertainty(
    model: Model,
    estimator: Estimator,
    input_text: str,
    input_image: Optional[Union[str, Path, Image.Image]] = None,
    available_stat_calculators=None
) -> UncertaintyOutput:
    if isinstance(model, WhiteboxModel):
        model_type = "Whitebox"
    elif isinstance(model, VisualWhiteboxModel):
        model_type = "VisualLM"
    else:
        model_type = "Blackbox"

    if available_stat_calculators is None and hasattr(estimator, "instructions"):
        from lm_polygraph.stat_calculators.ensemble_probs import EnsembleProbsCalculator
        from lm_polygraph.stat_calculators.input_texts import InputTextsCalculator
        from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator
        from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer

        class_labels = (
            getattr(estimator, "class_labels", None)
            or ["positive", "negative", "neutral"]
        )
        few_shot_examples = getattr(estimator, "few_shot_examples", None)
        prompt_formatting = getattr(estimator, "prompt_formatting", None)

        available_stat_calculators = [
            StatCalculatorContainer(
                name="input_texts",
                obj=InputTextsCalculator,
                builder="lm_polygraph.stat_calculators.input_texts",
                cfg={},
                dependencies=[],
                stats=["input_texts"],
            ),
            StatCalculatorContainer(
                name="ensemble_probs",
                obj=EnsembleProbsCalculator,
                builder="lm_polygraph.stat_calculators.ensemble_probs",
                cfg={
                    "instructions": estimator.instructions,
                    "class_labels": class_labels,
                    "few_shot_examples": few_shot_examples,
                    "prompt_formatting": prompt_formatting,
                },
                dependencies=["input_texts"],
                stats=["ensemble_probs"],
            ),
            StatCalculatorContainer(
                name="greedy_probs",
                obj=GreedyProbsCalculator,
                builder="lm_polygraph.stat_calculators.greedy_probs",
                cfg={},
                dependencies=["input_texts"],
                stats=["greedy_texts", "greedy_tokens", "greedy_log_probs", "greedy_probs"],
            ),
        ]

    man = UEManager(
        Dataset(
            [input_text],
            [""],
            batch_size=1,
            images=[input_image] if input_image is not None else None,
        ),
        model,
        [estimator],
        available_stat_calculators=available_stat_calculators or register_default_stat_calculators(
            model_type
        ),
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=False,
        max_new_tokens=model.generation_parameters.max_new_tokens,
    )
    man()
    ue = man.estimations[estimator.level, str(estimator)]
    texts = man.stats.get("greedy_texts", None)
    tokens = man.stats.get("greedy_tokens", None)
    if tokens is not None and len(tokens) > 0:
        tokens = tokens[0][:-1]
    return UncertaintyOutput(
        ue[0], input_text, texts[0], tokens, model.model_path, str(estimator)
    )