from lm_polygraph.stat_calculators import *


def load_stat_calculator(cfg, builder):
    sc = globals()[cfg["obj"]]()
    return sc
