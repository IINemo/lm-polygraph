from importlib import import_module
from lm_polygraph.stat_calculators import *
from typing import List

import logging

log = logging.getLogger()


def create_stat_calculator(module_name, config, environment):
    module = import_module(module_name)
    return module.load_stat_calculator(config, environment)


class StatCalculatorContainer:
    def __init__(
        self,
        name=None,
        stats=None,
        dependencies=None,
        obj=None,
        builder=None,
        cfg=dict(),
    ):
        self._name = name
        self.stats = stats if stats is not None else []
        self.dependencies = dependencies if dependencies is not None else []
        self.obj = obj
        self.cfg = cfg
        self.builder = builder

    @property
    def name(self):
        if self.obj is not None:
            return self.obj.__name__
        else:
            return self._name

    def meta_info(self):
        if self.obj is not None:
            return self.obj.meta_info()

        return self.stats, self.dependencies


class FactoryStatCalculator:
    def __init__(self, environment):
        self.environment = environment

    def __call__(self, stat_calculators_info: List[StatCalculatorContainer]):
        stat_calculators = [
            create_stat_calculator(
                sci.name if sci.builder is None else sci.builder,
                sci.cfg,
                self.environment,
            )
            for sci in stat_calculators_info
        ]
        return stat_calculators
