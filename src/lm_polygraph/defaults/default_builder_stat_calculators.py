from .factory_stat_calculator import FactoryStatCalculator


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


class DefaultBuilderStatCalculators:
    def __init__(self, model):
        self.model = model

    def __call__(self, stat_calculators_info):
        factory = FactoryStatCalculator()
        stat_calculators = [
            factory(sci.name if sci.builder is None else sci.builder, sci.cfg, self)
            for sci in stat_calculators_info
        ]
        return stat_calculators
