import os
<<<<<<< HEAD:src/lm_polygraph/utils/register_stat_calculators.py
import logging
=======
import inspect
>>>>>>> 585519b (Move stat dependency resolution to resolver):src/lm_polygraph/utils/stat_resolver.py

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.utils.openai_chat import OpenAIChat

from typing import Dict, List, Optional, Tuple, Set

log = logging.getLogger("lm_polygraph")


class StatResolver:
    def __init__(
        self,
        nli_model_batch_size: int = 10,
        nli_model_device: Optional[str] = None,
        cache_path: str = os.path.expanduser("~") + "/.cache"
    ):

        self.nli_model_batch_size = nli_model_batch_size
        self.nli_model_device = nli_model_device
        self.cache_path = cache_path
        self.nli_model = None
        self.openai_chat = None

        self.stat_calculators, self.stat_dependencies = \
            self._register_stat_calculators()

    def _register_stat_calculators(
        self
    ) -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
        """
        Registers all available statistic calculators to be seen by UEManager
        for properly organizing the calculations order.
        """
        stat_calculators: Dict[str, "StatCalculator"] = {}
        stat_dependencies: Dict[str, List[str]] = {}

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


    def order_stats(
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
                for new_stat in stat_calculators[stat].meta_info()[0]:
                    have_stats.add(new_stat)
        return ordered, have_stats


<<<<<<< HEAD:src/lm_polygraph/utils/register_stat_calculators.py
<<<<<<< HEAD
    log.info("Done intitializing stat calculators...")

    return stat_calculators, stat_dependencies
=======
    def init_calculators(self, 
=======
    def init_calculators(self, calculator_classes: List[StatCalculator]) -> List[StatCalculator]:
>>>>>>> 585519b (Move stat dependency resolution to resolver):src/lm_polygraph/utils/stat_resolver.py
        """
        Initializes all calculators needed for the given list of estimators
        """
        calculators = []
        for _class in calculator_classes:
            init_params = list(inspect.signature(_class.__init__).parameters.keys())
            args = {}
            if "nli_model" in init_params:
                # If the calculator needs the NLI model, we initialize it here
                # lazily to avoid unnecessary memory usage.
                # It will be reused by all calculators that need it.
                if self.nli_model is None:
                    self.nli_model = Deberta(
                        batch_size=self.nli_model_batch_size,
                        device=self.nli_model_device
                    )
                args["nli_model"] = self.nli_model
            if "openai_chat" in init_params: 
                # Same for OpenAI chat to reuse cache
                if self.openai_chat is None:
                    self.openai_chat = OpenAIChat(cache_path=self.cache_path)
                args["openai_chat"] = self.openai_chat

            calculators.append(_class(**args))

        return calculators
>>>>>>> 9098ef0 (WiP)
