import json
import sys
import os
import pathlib
import wget
from urllib.error import URLError

import numpy as np

from lm_polygraph.estimators import *

HOST = "http://209.38.249.180:8000"

DEFAULT_CACHE_PATH = f"{pathlib.Path(__file__).parent.resolve()}/normalization"


def normalization_bounds_present(
    est: Estimator,
    model_path: str,
    directory: str,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> bool:
    archive_path = model_path.split("/")[-1] + ".json"
    filepath = os.path.join(cache_path, archive_path)
    if os.path.exists(filepath):
        os.remove(filepath)
    try:
        wget.download(HOST + "/" + directory + "/" + archive_path, out=filepath)
    except URLError:
        sys.stderr.write("Warning: no normalization bounds found")
        return False
    with open(filepath, "r") as f:
        ue_bounds = json.load(f)
    return str(est) in ue_bounds.keys()


def can_get_calibration_conf(
    est: Estimator, model_path: str, cache_path: str = DEFAULT_CACHE_PATH
) -> bool:
    return normalization_bounds_present(
        est, model_path, "normalization_calib", cache_path
    )


def can_normalize_ue(
    est: Estimator, model_path: str, cache_path: str = DEFAULT_CACHE_PATH
) -> bool:
    return normalization_bounds_present(est, model_path, "normalization", cache_path)


def calibration_confidence(
    est: Estimator, model_path: str, val: float, cache_path: str = DEFAULT_CACHE_PATH
) -> float:
    if np.isnan(val):
        return 1
    est = str(est)
    filepath = os.path.join(cache_path, model_path.split("/")[-1] + ".json")
    with open(filepath, "r") as f:
        ue_bounds = json.load(f)
    if est not in ue_bounds.keys():
        sys.stderr.write(
            f"Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values."
        )
        return val

    ue_bins = ue_bounds[est]["ues"]
    conf_id = np.argwhere(np.array(ue_bins) > val).flatten()

    if len(conf_id) == 0:
        conf = ue_bounds[est]["normed_conf"][-1]
    elif conf_id[0] == 0:
        conf = ue_bounds[est]["normed_conf"][0]
    else:
        conf = ue_bounds[est]["normed_conf"][conf_id[0] - 1]

    return conf / 100


def normalize_ue(
    est: Estimator, model_path: str, val: float, cache_path: str = DEFAULT_CACHE_PATH
) -> float:
    if np.isnan(val):
        return 1
    filepath = os.path.join(cache_path, model_path.split("/")[-1] + ".json")
    with open(filepath, "r") as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f"Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values."
        )
        return val
    q = np.array(ue_bounds[str(est)])
    return (q < val).mean()
