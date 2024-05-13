from typing import List, Tuple, Dict

import numpy as np
from cir_model import CenteredIsotonicRegression
from scipy.stats import binned_statistic

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


def get_mans_ues_metrics(man_paths: List[str], ue_method_names: List[str], gen_metric_names: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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

    ues = concat_mans_data(mans_ues, ue_method_names)
    gen_metrics = concat_mans_data(mans_gen_metrics, gen_metric_names)

    return ues, gen_metrics


def filter_nans(gen_metrics: np.ndarray, ues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    ues = ues[~nan_mask]
    gen_metrics = gen_metrics[~nan_mask]

    return ues, gen_metrics


def fit_isotonic_pcc(gen_metrics: np.ndarray, ues: np.ndarray) -> CenteredIsotonicRegression:
    """Fits centered isotonic regression to the gen_metrics and ues data."""
    cir = CenteredIsotonicRegression(out_of_bounds='clip',
                                     increasing=False,
                                     y_min=0, y_max=1)
    cir.fit(ues, gen_metrics)

    return cir


def _get_bin_indices(ue, num_bins):
    per_bin = len(ue) / num_bins

    lesser_bin = math.floor(per_bin)
    greater_bin = math.ceil(per_bin)

    n_lesser_bins = (greater_bin * num_bins) - len(ue)
    n_greater_bins = num_bins - n_lesser_bins
    
    greater_start_index = n_lesser_bins * lesser_bin

    bin_indices = [i * lesser_bin for i in range(n_lesser_bins)] + \
                  [greater_start_index + i * greater_bin for i in range(n_greater_bins)] + \
                  [len(ue) - 1]

    return bin_indices


def _get_bin_edges(ue, num_bins):
    sorted_ue = np.sort(ue)
    bin_indices = _get_bin_indices(sorted_ue, num_bins)
    bin_edges = np.unique([sorted_ue[i] for i in bin_indices]).tolist()

    return bin_edges


def _get_bins(ue, metric, edges):
    metric_bins = binned_statistic(ue,
                                   metric,
                                   bins=edges,
                                   statistic='mean')

    return metric_bins


def fit_binned_pcc(gen_metrics: np.ndarray,
                   ues: np.ndarray, num_bins: int) -> CenteredIsotonicRegression:
    bin_edges = _get_bin_edges(ues, num_bins)
    metric_bins, std_bins, sem_bins = _get_bins(ues, gen_metrics, bin_edges)

    binned_metric = metric_bins.statistic

    return {
        "binned_metric": binned_metric,
        "bin_edges": bin_edges
    }


def fit_quantile(ues: np.ndarray) -> QuantileTransformer:
    """Fits QuantileTransformer to the gen_metrics and ues data."""
    scaler = QuantileTransformer(output_distribution='uniform')
    scaler.fit(ues)

    return scaler


def fit_min_max(ues: np.ndarray) -> MinMaxScaler:
    """Fits MinMaxScaler to the gen_metrics and ues data."""
    scaler = MinMaxScaler(clip=True)
    scaler.fit(ues)

    return scaler
