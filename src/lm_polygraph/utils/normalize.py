import json
import sys
import os
import pathlib
import wget

import numpy as np

from lm_polygraph.estimators import *

HOST = 'http://209.38.249.180:8000'

DEFAULT_CACHE_PATH = f'{pathlib.Path(__file__).parent.resolve()}/normalization'


def can_normalize_ue(est: Estimator, model_path: str, cache_path: str = DEFAULT_CACHE_PATH) -> bool:
    archive_path = model_path.split('/')[-1] + '.json'
    filepath = os.path.join(cache_path, archive_path)
    if not os.path.exists(filepath):
        sys.stdout.write(f'No cached normalizing bounds for estimator: {str(est)}. Looking in remote storage...')
        try:
            wget.download(HOST + '/normalization/' + archive_path, out = filepath)
        except:
            sys.stderr.write('Failed, no normalization...')
            return False
    with open(filepath, 'r') as f:
        ue_bounds = json.load(f)
    return str(est) in ue_bounds.keys()


def normalize_ue(est: Estimator, model_path: str, val: float, cache_path: str = DEFAULT_CACHE_PATH) -> float:
    if np.isnan(val):
        return 1
    filepath = os.path.join(cache_path, model_path.split('/')[-1] + '.json')
    with open(filepath, 'r') as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f'Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values.')
        return val
    q = np.array(ue_bounds[str(est)])
    return (q < val).mean()
