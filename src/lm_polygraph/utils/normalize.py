import json
import sys
import os
import pathlib

import numpy as np

from lm_polygraph.estimators import *

NORMALIZATION_PATH = f'{pathlib.Path(__file__).parent.resolve()}/normalization'


def can_normalize_ue(est: Estimator, model_path: str) -> bool:
    filepath = os.path.join(NORMALIZATION_PATH, model_path.split('/')[-1] + '.json')
    if not os.path.exists(filepath):
        return False
    with open(filepath, 'r') as f:
        ue_bounds = json.load(f)
    return str(est) in ue_bounds.keys()


def normalize_ue(est: Estimator, model_path: str, val: float) -> float:
    if np.isnan(val):
        return 1
    filepath = os.path.join(NORMALIZATION_PATH, model_path.split('/')[-1] + '.json')
    with open(filepath, 'r') as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f'Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values.')
        return val
    q = np.array(ue_bounds[str(est)])
    return (q < val).mean()
