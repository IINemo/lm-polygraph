import json
import sys
import pathlib

import numpy as np

from lm_polygraph.estimators import *

UE_BOUNDS_FILEPATH = f'{pathlib.Path(__file__).parent.resolve()}/ue_bounds.json'


def has_norm_bound(est: Estimator) -> bool:
    with open(UE_BOUNDS_FILEPATH, 'r') as f:
        ue_bounds = json.load(f)
    return str(est) in ue_bounds.keys()


def normalize_from_bounds(est: Estimator, val: float) -> float:
    if np.isnan(val):
        return 1
    with open(UE_BOUNDS_FILEPATH, 'r') as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f'Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values.')
        return val
    low, high = ue_bounds[str(est)]['low'], ue_bounds[str(est)]['high']
    return min(1, max(0, (val - low) / (high - low)))
