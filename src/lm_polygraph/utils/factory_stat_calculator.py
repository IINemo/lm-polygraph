from importlib import import_module
from lm_polygraph.stat_calculators import *

import logging
log = logging.getLogger()


def load_simple_stat_calculator(name, config):
    SIMPLE_STAT_CALCULATORS = [
        GreedyLMProbsCalculator,
        EntropyCalculator,
        BartScoreCalculator,
        EmbeddingsCalculator,
        EnsembleTokenLevelDataCalculator,
        GreedyProbsCalculator,
        InferCausalLMCalculator,
        ModelScoreCalculator,
        BasePromptCalculator,
        BlackboxSamplingGenerationCalculator
    ]

    try:
        simple_stat_calculators = {e.__name__: e for e in SIMPLE_STAT_CALCULATORS}
        sc = simple_stat_calculators[name](**config)
        return sc
    
    except KeyError:
        return None


class FactoryStatCalculator:
    def __call__(self, name, config, builder):
        est = load_simple_stat_calculator(name, config)
        if est is not None:
            return est
        
        log.info(f"Loading stat calculator {name}")
        module = import_module(name)
        return module.load_stat_calculator(config, builder)
