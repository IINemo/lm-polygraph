from lm_polygraph.stat_calculators import *
from .builder_enviroment_stat_calculator import BuilderEnvironmentBase
import logging

log = logging.getLogger(__name__)


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    sc = globals()[cfg.obj]()
    return sc
