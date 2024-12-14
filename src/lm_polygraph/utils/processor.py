import numpy as np

import logging
from typing import List, Dict, Tuple
from lm_polygraph.estimators.estimator import Estimator

log = logging.getLogger(__name__)


class Processor:
    """
    Abstract class to perform actions after processing new texts batch.
    """

    def on_batch(
        self,
        batch_stats: Dict[str, np.ndarray],
        batch_gen_metrics: Dict[Tuple[str, str], List[float]],
        batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        """
        Processes new batch.

        Parameters:
            batch_stats (Dict[str, np.ndarray]): Dictionary of statistics calculated with `stat_calculators`.
            batch_gen_metrics (Dict[Tuple[str, str], List[float]]): Dictionary of generation metrics calculated
                for the batch. Dictionary keys consist of UE level (`sequence` or `token`) and generation metrics
                name.
            batch_estimations (Dict[Tuple[str, str], List[float]]): Dictionary of UE estimations calculated
                for the batch. Dictionary keys consist of UE level (`sequence` or `token`) and UE estimator name.
        """
        pass

    def on_eval(self, metrics: Dict[Tuple[str, str, str, str], float]):
        """
        Processes newly calculated evaluation metrics.

        Parameters:
            metrics (Dict[Tuple[str, str, str, str], float]: metrics calculated using `ue_metrics` on the batch which
                was considered at the last `on_batch` call. Dictionary keys consist of UE level,
                estimator name, generation metrics name and `ue_metrics` name which was used to calculate quality
                metrics between this estimator's uncertainty estimations and generation metric outputs.
        """
        pass


class Logger(Processor):
    """
    Processor logging batch information to stdout.
    """

    def on_batch(
        self,
        batch_stats: Dict[str, np.ndarray],
        batch_gen_metrics: Dict[Tuple[str, str], List[float]],
        batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        """
        Outputs statistics from `batch_stats`, `batch_gen_metrics` and `batch_estimations` to stdout.
        """
        log.info("=" * 50 + " NEW BATCH " + "=" * 50)
        log.info("Statistics:")
        log.info("")
        for key, val in batch_stats.items():
            str_repr = str(val)
            # to skip large outputs
            if len(str_repr) < 10000 and str_repr.count("\n") < 10:
                log.info(f"{key}: {val}")
                log.info("")
        log.info("-" * 100)
        log.info("Estimations:")
        log.info("")
        for key, val in batch_estimations.items():
            log.info(f"{key}: {val}")
            log.info("")
        log.info("-" * 100)
        log.info("Generation metrics:")
        log.info("")
        for key, val in batch_gen_metrics.items():
            log.info(f"{key}: {val}")
            log.info("")

    def on_eval(
        self,
        metrics: Dict[Tuple[str, str, str, str], float],
        bad_estimators: Dict[Estimator, int],
    ):
        """
        Outputs statistics from `metrics` and failed estimators to stdout.
        """
        log.info("=" * 50 + " METRICS " + "=" * 50)
        log.info("Metrics:")
        log.info("")
        for key, val in metrics.items():
            log.info(f"{key}: {val}")
            log.info("")
        if len(bad_estimators) > 0:
            log.info("=" * 45 + " FAILED ESTIMATORS " + "=" * 45)
            for bad_estimator, batch_i in bad_estimators.items():
                log.info(str(bad_estimator) + " on batch " + str(batch_i))
