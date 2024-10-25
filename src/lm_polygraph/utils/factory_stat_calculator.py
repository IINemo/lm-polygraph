from importlib import import_module
from lm_polygraph.stat_calculators import *

import logging

log = logging.getLogger()


class FactoryStatCalculator:
    def __call__(self, name, config, builder):
        # est = load_simple_stat_calculator(name, config)
        # if est is not None:
        #     return est

        # log.info(f"Trying to load stat calculator {name}")
        # loader = get_stat_calculator_loader(name)
        # if loader is not None:
        #     log.info(f"Found a default loader {loader}.")
        #     name = loader
        module = import_module(name)
        print(config)
        return module.load_stat_calculator(config, builder)
