import lm_polygraph.stat_calculators as calculators
from .builder_enviroment_stat_calculator import BuilderEnvironmentBase
import logging

log = logging.getLogger(__name__)


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    sc = getattr(calculators, cfg.obj)()
    return sc
