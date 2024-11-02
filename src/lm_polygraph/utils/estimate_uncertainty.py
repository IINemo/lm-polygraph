from typing import List, Set, Dict, Union
from dataclasses import dataclass

from lm_polygraph.utils.model import Model, WhiteboxModel
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
    """
    Uncertainty estimator output.

    Parameters:
        uncertainty (float): uncertainty estimation.
        input_text (str): text used as model input.
        generation_text (str): text generated by the model.
        model_path (str): path to the model used in generation.
    """

    uncertainty: Union[float, List[float]]
    input_text: str
    generation_text: str
    generation_tokens: List[int]
    model_path: str
    estimator: str


def estimate_uncertainty(
    model: Model, estimator: Estimator, input_text: str
) -> UncertaintyOutput:
    """
    Estimated uncertainty of the model generation using the provided esitmator.

    Parameters:
        model (Model): model to estimate uncertainty of. Either lm_polygraph.WhiteboxModel or
            lm_polygraph.BlackboxModel model can be used.
        estimator (Estimator): uncertainty estimation method to use. Can be any of the methods at
            lm_polygraph.estimators.
        input_text (str): text to estimate uncertainty of.
    Returns:
        UncertaintyOutput: uncertainty estimation float along with supporting info.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.estimators import LexicalSimilarity
    >>> model = WhiteboxModel.from_pretrained(
    ...     'bigscience/bloomz-560m',
    ...     device='cpu',
    ... )
    >>> estimator = LexicalSimilarity('rougeL')
    >>> estimate_uncertainty(model, estimator, input_text='Who is George Bush?')
    UncertaintyOutput(uncertainty=-0.9176470588235295, input_text='Who is George Bush?', generation_text=' President of the United States', model_path='bigscience/bloomz-560m')
    ```

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> from lm_polygraph.estimators import EigValLaplacian
    >>> model = BlackboxModel.from_openai(
    ...     'YOUR_OPENAI_TOKEN',
    ...     'gpt-3.5-turbo'
    ... )
    >>> estimator = EigValLaplacian()
    >>> estimate_uncertainty(model, estimator, input_text='When did Albert Einstein die?')
    UncertaintyOutput(uncertainty=1.0022274826855433, input_text='When did Albert Einstein die?', generation_text='Albert Einstein died on April 18, 1955.', model_path='gpt-3.5-turbo')
    ```
    """
    model_type = "Whitebox" if isinstance(model, WhiteboxModel) else "Blackbox"
    man = UEManager(
        Dataset([input_text], [""], batch_size=1),
        model,
        [estimator],
        available_stat_calculators=register_default_stat_calculators(
            model_type
        ),  # TODO:
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=False,
    )
    man()
    ue = man.estimations[estimator.level, str(estimator)]
    texts = man.stats.get("greedy_texts", None)
    tokens = man.stats.get("greedy_tokens", None)
    if tokens is not None and len(tokens) > 0:
        # Remove last token, which is the end of the sequence token
        # since we don't include it's uncertainty in the estimator's output
        tokens = tokens[0][:-1]
    return UncertaintyOutput(
        ue[0], input_text, texts[0], tokens, model.model_path, str(estimator)
    )
