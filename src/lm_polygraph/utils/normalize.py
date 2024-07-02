from typing import List, Tuple, Dict

import numpy as np

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.common import seq_man_key


def _concat_mans_data(mans_data_dicts, names):
    """Concatenates data from multiple manager data dictionaries.

    Args:
    mans_data_dicts: List of dictionaries, where each dictionary contains
      data of particular type from a single manager.
      Each dictionary should have the same keys.
    names: List of value types to extract from the dictionaries.

    Returns:
    Dictionary, where keys are the input names and values are concatenated
    arrays of the data from all managers.
    """
    data = {}
    for name in names:
        man_data = []
        for man_data_dict in mans_data_dicts:
            key = seq_man_key(name)
            try:
                man_data.append(man_data_dict[key])
            except KeyError:
                raise KeyError(f"{key} not found in manager data")
        data[name] = np.concatenate(man_data)

    return data


def get_mans_ues_metrics(
    man_paths: List[str], ue_method_names: List[str], gen_metric_names: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extracts and concats data from a list of paths to
    saved manager data files.

    Args:
    man_paths: List of paths to manager data files
    ue_method_names: List of UE methods to extract
    gen_metric_names: List of gen_metrics to extract

    Returns:
    Tuple of two dictionaries:
    - First dictionary contains UE method data, where keys are method names
      and values are concatenated arrays of UE method data from all managers
    - Second dictionary contains gen_metric data, where keys are metric names
      and values are concatenated arrays of gen_metric data from all managers
    """

    mans = [UEManager.load(path) for path in man_paths]
    mans_ues = [man.estimations for man in mans]
    mans_gen_metrics = [man.gen_metrics for man in mans]

    ues = _concat_mans_data(mans_ues, ue_method_names)
    gen_metrics = _concat_mans_data(mans_gen_metrics, gen_metric_names)

    return ues, gen_metrics


def filter_nans(
    gen_metrics: np.ndarray, ues: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters out NaNs from gen_metrics and ues if they occur at least
    in one of the arrays.

    Args:
    gen_metrics: Array of gen_metrics
    ues: Array of ues

    Returns:
    Tuple of two arrays:
    - First array contains gen_metrics with NaNs removed
    - Second array contains ues with NaNs removed
    """

    nan_mask = np.isnan(gen_metrics) | np.isnan(ues)
    gen_metrics = gen_metrics[~nan_mask]
    ues = ues[~nan_mask]

    return gen_metrics, ues
