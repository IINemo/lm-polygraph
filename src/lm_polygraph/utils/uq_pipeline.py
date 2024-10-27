from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import sys
import traceback
from collections import defaultdict
import torch
import gc

from lm_polygraph.utils.common import flatten_results


class UQPipeline:
    def __init__(self, stat_calculators, estimators):
        self.stat_calculators = stat_calculators
        self.estimators = estimators

    def __call__(
        self, iterable_data, batch_callback
    ) -> Dict[Tuple[str, str, str, str], float]:
        """
        Runs benchmark and reports metrics results. Saves all useful calculated statistics for further usage.
        The run includes:
        * Calculating uncertainty estimations for each `estimator` for all input texts in the dataset
        * Calculating ground-truth uncertainties for each `generation_metrics` for all input texts in the dataset.
        * Calculating correlation measure for each `ue_metrics`, between each pair of
        (uncertainty estimation, ground-truth uncertainty) which comes from the same level
        (both 'sequence' or both 'token').
        * Saving uncertainty estimations, ground-truth uncertainties and ue_metrics values for further usage.

        Returns:
            [Tuple[str, str, str, str], float]: dictionary with metrics results. Dictionary keys consist of
                - uncertainty estimation level: 'sequence' or 'token',
                - estimator name,
                - generation metrics name,
                - `ue_metrics` name which was used to calculate quality.
        """

        iterable_data = tqdm(self.data) if self.verbose else self.data
        for batch_i, (inp_texts, target_texts) in enumerate(iterable_data):
            batch_stats: Dict[str, np.ndarray] = {}
            for key, val in [
                ("input_texts", inp_texts),
                ("target_texts", target_texts),
            ]:
                self.stats[key] += val
                batch_stats[key] = val
            batch_stats["model"] = self.model

            batch_stats = self.calculate(batch_stats, self.stat_calculators, inp_texts)

            batch_estimations, bad_estimators = self.estimate(
                batch_stats, self.estimators
            )

            batch_callback(batch_i, target_texts, batch_stats, batch_estimations, bad_estimators)

            torch.cuda.empty_cache()
            gc.collect()

        return self.estimations        


    def calculate(self, batch_stats: dict, calculators: list, inp_texts: list) -> dict:
        """
        Runs stat calculators and handles errors if any occur. Returns updated batch stats

        Parameters:
            batch_stats (dict): contains current batch statistics to be updated
            calculators (list): list of stat calculators to run
            inp_texts (list): list of inputs to the model in the batch
        """
        for stat_calculator in calculators:
            try:
                new_stats = stat_calculator(
                    batch_stats, inp_texts, self.model, self.max_new_tokens
                )
                for stat, stat_value in new_stats.items():
                    if stat in batch_stats.keys():
                        continue
                    batch_stats[stat] = stat_value
                    if (f"blackbox_{stat}" in self.stat_calculators_dict.keys()) and (
                        f"blackbox_{stat}" in self.stats_names
                    ):
                        batch_stats[f"blackbox_{stat}"] = stat_value
            except Exception as e:
                if self.ignore_exceptions:
                    lineno = e.__traceback__.tb_lineno
                    log_msg = f"Caught exception while calculating stats: {e} in Stat Calculator {stat_calculator}, line {lineno}. Expect dependent estimator to fail.\n"
                    sys.stderr.write("\n\n")
                    sys.stderr.write(log_msg)
                    sys.stderr.write(traceback.format_exc())
                    continue
                else:
                    raise e

        return batch_stats

    def estimate(
        self, batch_stats: dict, estimators: list
    ) -> Dict[Tuple[str, str], List[float]]:
        """
        Runs stat calculators and handles errors if any occur. Returns updated batch stats

        Parameters:
            batch_stats (dict): contains current batch statistics to be updated
            estimators (list): list of estimators to run
        """
        batch_estimations = defaultdict(list)
        bad_estimators = []

        for estimator in estimators:
            try:
                e = estimator(batch_stats)
                if not isinstance(e, list):
                    e = e.tolist()
                if estimator.level == "claim":
                    e = flatten_results(e, estimator)
                self.estimations[estimator.level, str(estimator)] += e
                batch_estimations[estimator.level, str(estimator)] += e
            except Exception as e:
                if self.ignore_exceptions:
                    bad_estimators.append(estimator)
                    lineno = e.__traceback__.tb_lineno
                    log_msg = f"Caught exception while estimating uncertainty: {e} in estimator {estimator}, line {lineno}. Estimator will be removed.\n"
                    sys.stderr.write("\n\n")
                    sys.stderr.write(log_msg)
                    sys.stderr.write(traceback.format_exc())
                    continue
                else:
                    raise e

        return batch_estimations, bad_estimators
