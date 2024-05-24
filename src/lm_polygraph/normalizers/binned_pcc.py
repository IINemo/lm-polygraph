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
        """Given an array of unceratinty estimates, returns the
        indices of the elements that should be used to create the bins.
        Bins are created such that each bin has approximately the same number of
        elements.
        """
        # Calculate the number of elements in each bin
        per_bin = len(ue) / num_bins

        # If the number of elements is not divisible by the number of bins,
        # then the number of elements in each bin will not be exactly equal.
        # If it is divisible, then lesser_bin and greater_bin will be the same.
        lesser_bin = math.floor(per_bin)
        greater_bin = math.ceil(per_bin)

        # If we use greater_bin for all bins, then the number of elements in the bins
        # will be greater than the number of elements in the original array.
        # Number of excess elements is the number of bins that should have
        # one less element in them, i.e. n_lesser_bins.
        n_lesser_bins = (greater_bin * num_bins) - len(ue)
        n_greater_bins = num_bins - n_lesser_bins

        # Calculate total number of elements in lesser bins. This is the index
        # where the greater bins start.
        greater_start_index = n_lesser_bins * lesser_bin

        # Calculate the indices of the elements that should be used to create the bins.
        # First we create the indices for the lesser bins, then the greater bins, and
        # finally the last element.
        bin_indices = (
            [i * lesser_bin for i in range(n_lesser_bins)]
            + [greater_start_index + i * greater_bin for i in range(n_greater_bins)]
            + [len(ue) - 1]
        )

        return bin_indices

    def _get_bin_edges(self, ue, num_bins):
        """Given an array of unceratinty estimates, returns the bin edges
        for the given number of bins.
        Bins are created such that each bin has approximately the same number of
        elements.
        """
        bin_indices = self._get_bin_indices(ue, num_bins)
        # Get the unique values of the uncertainty estimates at the bin indices
        sorted_ue = np.sort(ue)
        bin_edges = np.unique([sorted_ue[i] for i in bin_indices]).tolist()

        return bin_edges

    def _get_bins(self, ue, metric, edges):
        metric_bins = binned_statistic(ue, metric, bins=edges, statistic="mean")

        return metric_bins

    def fit(self, gen_metrics: np.ndarray, ues: np.ndarray, num_bins: int) -> None:
        """Fits BinnedPCCNormalizer to the gen_metrics and ues data."""
        # Get the bin edges corresponding to the number of bins that have
        # approximately the same number of elements in each bin.
        bin_edges = self._get_bin_edges(ues, num_bins)

        # Get average metric values for each bin
        metric_bins = self._get_bins(ues, gen_metrics, bin_edges)
        binned_metric = metric_bins.statistic

        self.params = {"binned_metric": binned_metric, "bin_edges": bin_edges}

    def transform(self, ues: np.ndarray) -> np.ndarray:
        """Transforms the ues data using the fitted BinnedPCCNormalizer."""
        bins = np.array(self.params["bin_edges"])
        calibrated_ues = []
        for ue in ues:
            # Find the bin in which the uncertainty estimate falls
            calibration_bin = np.argmax(bins >= ue) - 1
            # Calibrated confidence value is the average metric value in the bin
            calibrated_ues.append(self.params["binned_metric"][calibration_bin])

        return np.array(calibrated_ues)

    def dumps(self) -> str:
        """Dumps params of a BinnedPCCNormalizer object to a string."""
        return pickle.dumps(self.params)

    @staticmethod
    def loads(encoded_params: str):
        """Loads the BinnedPCCNormalizer object from a string of parameters."""
        normalizer = BinnedPCCNormalizer()
        normalizer.params = pickle.loads(encoded_params)

        return normalizer
