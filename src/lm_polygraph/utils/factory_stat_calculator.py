from importlib import import_module
from typing import List

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentBase

import logging

log = logging.getLogger()


def create_stat_calculator(module_name: str, config, environment):
    module = import_module(module_name)
    return module.load_stat_calculator(config, environment)


class StatCalculatorContainer:
    """The description of a stat calculator that is used to build the stat calculator."""

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
    """Constructs a stat calculator from a given name and configuration."""

    def __init__(self, environment: BuilderEnvironmentBase):
        self.environment = environment

    def __call__(
        self, stat_calculators_info: List[StatCalculatorContainer]
    ) -> List[StatCalculator]:
        stat_calculators = [
            create_stat_calculator(
                sci.name if sci.builder is None else sci.builder,
                sci.cfg,
                self.environment,
            )
            for sci in stat_calculators_info
        ]
        return stat_calculators
