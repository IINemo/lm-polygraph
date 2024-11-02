from abc import ABC


class BuilderEnvironmentBase(ABC):
    pass


class BuilderEnvironmentStatCalculator(BuilderEnvironmentBase):
    """Environment seen by all stat calculators when they are built in polygraph_eval script. Stat calculators can share the constructed objects via the environment."""

    def __init__(self, model):
        self.model = model
