import os
import logging

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.utils.openai_chat import OpenAIChat

from typing import Dict, List, Optional, Tuple

log = logging.getLogger("lm_polygraph")


class Resolver:
    def __init__(self):
        self.stat_calculators, self.stat_dependencies = \
            self.register_stat_calculators()

    def _register_stat_calculators(
        self,
        deberta_batch_size: int = 10,  # TODO: rename to NLI model
        deberta_device: Optional[str] = None,  # TODO: rename to NLI model
        n_ccp_alternatives: int = 10,
        cache_path=os.path.expanduser("~") + "/.cache",
    ) -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
        """
        Registers all available statistic calculators to be seen by UEManager for properly organizing the calculations
        order.
        """
        stat_calculators: Dict[str, "StatCalculator"] = {}
        stat_dependencies: Dict[str, List[str]] = {}
>>>>>>> 9098ef0 (WiP)

        log.info("=" * 100)
        log.info("Loading NLI model...")
        #nli_model = Deberta(batch_size=deberta_batch_size, device=deberta_device)
        nli_model = None
        openai_chat = OpenAIChat(cache_path=cache_path)

        log.info("=" * 100)
        log.info("Initializing stat calculators...")

        def _register(calculator_class: StatCalculator):
            stats, dependencies = calculator_class.meta_info()
            for stat in stats:
                if stat in stat_calculators.keys():
                    continue
                stat_calculators[stat] = calculator_class
                stat_dependencies[stat] = dependencies

        _register(GreedyProbsCalculator)
        _register(BlackboxGreedyTextsCalculator)
        _register(EntropyCalculator)
        _register(GreedyLMProbsCalculator)
        _register(PromptCalculator)
        _register(SamplingPromptCalculator)
        _register(ClaimPromptCalculator)
        _register(SamplingGenerationCalculator)
        _register(BlackboxSamplingGenerationCalculator)
        _register(BartScoreCalculator)
        _register(ModelScoreCalculator)
        _register(EmbeddingsCalculator)
        _register(EnsembleTokenLevelDataCalculator)
        _register(SemanticMatrixCalculator)
        _register(CrossEncoderSimilarityMatrixCalculator)
        _register(GreedyProbsCalculator)
        _register(GreedyAlternativesNLICalculator)
        _register(GreedyAlternativesFactPrefNLICalculator)
        _register(ClaimsExtractor)

        return stat_calculators, stat_dependencies


    def _order_calculators(
        self,
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
                for new_stat in stat_calculators[stat].stats:
                    have_stats.add(new_stat)
        return ordered, have_stats


<<<<<<< HEAD
    log.info("Done intitializing stat calculators...")

    return stat_calculators, stat_dependencies
=======
    def init_calculators(self, 
        """
        Initializes all calculators needed for the given list of estimators
        """
        ordered, have_stats = self._order_calculators(
            stats, self.stat_calculators, self.stat_dependencies
        )
        calculators = []
        for stat in ordered:
            calculators.append(self.stat_calculators[stat])
        return calculators
>>>>>>> 9098ef0 (WiP)
