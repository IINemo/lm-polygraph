from lm_polygraph.stat_calculators import *
from .builder_enviroment_stat_calculator import BuilderEnvironmentBase


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    sc = globals()[cfg.obj]()
    return sc
