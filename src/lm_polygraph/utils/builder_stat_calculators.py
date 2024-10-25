from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, Model

from typing import List, Set, Dict, Tuple, Optional, Union

import logging

log = logging.getLogger()


def order_stats(
    stats: List[str],
    stat_calculators: Dict[str, StatCalculator],
    stat_dependencies: Dict[str, List[str]],
) -> Tuple[List[str], Set[str]]:
    ordered: List[str] = []
    have_stats: Set[str] = set()
    while len(stats) > 0:
        stat = stats[0]
        if stat in have_stats:
            stats = stats[1:]
            continue
        dependent = False
        if stat not in stat_dependencies.keys():
            raise Exception(
                f"Cant find stat calculator for: {stat}. Maybe you forgot to register it in "
                + "lm_polygraph.utils.register_stat_calculators.register_stat_calculators()?"
            )
        for d in stat_dependencies[stat]:
            if d not in have_stats:
                stats = [d] + stats
                if stats.count(d) > 40:
                    raise Exception(f"Found possibly cyclic dependencies: {d}")
                dependent = True
        if not dependent:
            stats = stats[1:]
            ordered.append(stat)
            for new_stat in stat_calculators[stat].meta_info()[0]:
                have_stats.add(new_stat)

    return ordered, have_stats


class BuilderStatCalculators:
    def __init__(self, available_stat_calculators, builder_environment):
        self._builder_environment = builder_environment
        self._available_stat_calculators = available_stat_calculators

    def _resolve_stat_calculators(self):
        log.info("=" * 100)
        log.info("Initializing stat calculators...")

        stat_calculators_dict = self.stat_calculators_dict
        stat_dependencies_dict = self.stat_dependencies_dict

        # self.stat_calculators_dict = stat_calculators_dict

        # if isinstance(self.model, BlackboxModel):
        #     greedy = ["blackbox_greedy_texts"]
        # else:
        #     greedy = ["greedy_tokens", "greedy_texts"]

        greedy = ["greedy_texts"]
        if not isinstance(self.model, BlackboxModel):
            greedy += ["greedy_tokens"]

        stats = (
            [s for e in self.estimators for s in e.stats_dependencies]
            + [s for m in self.generation_metrics for s in m.stats_dependencies]
            + greedy
        )

        stats, have_stats = order_stats(
            stats,
            stat_calculators_dict,
            stat_dependencies_dict,
        )

        self.stats_names = stats
        stats = [
            s
            for s in stats
            if not (str(s).startswith("ensemble_"))
            and not (
                (
                    str(s).startswith("blackbox_")
                    and s[len("blackbox_") :] in have_stats
                )  # remove blackbox_X from stats only if X is already in stats to remove duplicated run of stat calculator
            )
        ]  # below in calculate() we copy X in blackbox_X

        self.stat_calculators = self._builder_environment(
            [stat_calculators_dict[c] for c in stats]
        )

        # : List[StatCalculator] = (
        #     self.stat_resolver.init_calculators(
        #         [stat_calculators_dict[c] for c in stats]
        #     )
        # )
        if self.verbose:
            print("Stat calculators:", self.stat_calculators)

        self.ensemble_estimators = []
        single_estimators = []
        for e in self.estimators:
            for s in e.stats_dependencies:
                if s.startswith("ensemble"):
                    self.ensemble_estimators.append(e)
                    break
            if e not in self.ensemble_estimators:
                single_estimators.append(e)
        self.estimators = single_estimators

        train_stats = [
            s
            for e in self.estimators
            for s in e.stats_dependencies
            if s.startswith("train")
        ]
        train_stats += (
            ["greedy_tokens", "greedy_texts"]
            if "train_greedy_log_likelihoods" in train_stats
            else []
        )
        train_stats, _ = order_stats(
            train_stats,
            stat_calculators_dict,
            stat_dependencies_dict,
        )
        # self.train_stat_calculators: List[StatCalculator] = (
        #     self.stat_resolver.init_calculators(
        #         [stat_calculators_dict[c] for c in train_stats]
        #     )
        # )
        self.train_stat_calculators = self.builder_stat_calculators(train_stats)

        background_train_stats = [
            s
            for e in self.estimators
            for s in e.stats_dependencies
            if s.startswith("background_train")
        ]
        background_train_stats, _ = order_stats(
            background_train_stats,
            stat_calculators_dict,
            stat_dependencies_dict,
        )

        self.background_train_stat_calculators = self.builder_stat_calculators(
            background_train_stats
        )

        # self.background_train_stat_calculators: List[StatCalculator] = (
        #     self.stat_resolver.init_calculators(
        #         [stat_calculators_dict[c] for c in background_train_stats]
        #     )
        # )

        ensemble_stats = [
            s
            for e in self.ensemble_estimators
            for s in e.stats_dependencies
            if s.startswith("ensemble")
        ]
        ensemble_stats, _ = order_stats(
            ensemble_stats,
            stat_calculators_dict,
            stat_dependencies_dict,
        )
        self.ensemble_stat_calculators = self.builder_stat_calculators(ensemble_stats)
        # self.ensemble_stat_calculators: List[StatCalculator] = (
        #     self.stat_resolver.init_calculators(
        #         [stat_calculators_dict[c] for c in ensemble_stats]
        #     )
        # )

        log.info("Done intitializing stat calculators...")
