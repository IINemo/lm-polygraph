import json
import sys

import numpy as np

from estimators import *

UE_BOUNDS_FILEPATH = 'utils/ue_bounds.json'


def normalize_ue(est: Estimator, val: float) -> float:
    if np.isnan(val):
        return 1
    with open(UE_BOUNDS_FILEPATH, 'r') as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f'Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values.')
        return val
    low, high = ue_bounds[str(est)]['low'], ue_bounds[str(est)]['high']
    return min(1, max(0, (high - val) / (high - low)))
