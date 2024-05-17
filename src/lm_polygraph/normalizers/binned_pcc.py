# Description: BinnedPCC normalizer for UE values

import pickle
import math

import numpy as np
from scipy.stats import binned_statistic

from lm_polygraph.normalizers.base import BaseUENormalizer


class BinnedPCCNormalizer(BaseUENormalizer):
    def __init__(self):
        self.params = None

    def _get_bin_indices(self, ue, num_bins):
        per_bin = len(ue) / num_bins

        lesser_bin = math.floor(per_bin)
        greater_bin = math.ceil(per_bin)

        n_lesser_bins = (greater_bin * num_bins) - len(ue)
        n_greater_bins = num_bins - n_lesser_bins

        greater_start_index = n_lesser_bins * lesser_bin

        bin_indices = (
            [i * lesser_bin for i in range(n_lesser_bins)]
            + [greater_start_index + i * greater_bin for i in range(n_greater_bins)]
            + [len(ue) - 1]
        )

        return bin_indices

    def _get_bin_edges(self, ue, num_bins):
        sorted_ue = np.sort(ue)
        bin_indices = self._get_bin_indices(sorted_ue, num_bins)
        bin_edges = np.unique([sorted_ue[i] for i in bin_indices]).tolist()

        return bin_edges

    def _get_bins(self, ue, metric, edges):
        metric_bins = binned_statistic(ue, metric, bins=edges, statistic="mean")

        return metric_bins

    def fit(self, gen_metrics: np.ndarray, ues: np.ndarray, num_bins: int) -> None:
        """Fits BinnedPCCNormalizer to the gen_metrics and ues data."""

        bin_edges = self._get_bin_edges(ues, num_bins)
        metric_bins = self._get_bins(ues, gen_metrics, bin_edges)

        binned_metric = metric_bins.statistic

        self.params = {"binned_metric": binned_metric, "bin_edges": bin_edges}

    def transform(self, ues: np.ndarray) -> np.ndarray:
        """Transforms the ues data using the fitted BinnedPCCNormalizer."""
        bins = np.array(self.params["bin_edges"])
        calibrated_ues = []
        for ue in ues:
            calibration_bin = np.argmax(bins >= ue) - 1
            calibrated_ues.append(self.params["binned_metric"][calibration_bin])

        return np.array(calibrated_ues)

    def dumps(self) -> str:
        """Dumps params of a BinnedPCCNormalizer object to a string."""
        return pickle.dumps(self.params)

    @staticmethod
    def loads(scaler):
        """Loads the BinnedPCCNormalizer object from a string of parameters."""
        normalizer = BinnedPCCNormalizer()
        normalizer.params = pickle.loads(scaler)

        return normalizer
