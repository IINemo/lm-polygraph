from abc import ABC


class BuilderEnvironmentBase(ABC):
    pass


class BuilderEnvironmentStatCalculator(BuilderEnvironmentBase):
    def __init__(self, model):
        self.model = model
