import numpy as np
import torch
import sys
import os

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional
from tqdm import tqdm

from utils.dataset import Dataset
from utils.model import Model
from utils.ensemble_generator import EnsembleGenerator
from utils.processor import Processor
from generation_metrics.generation_metric import GenerationMetric
from ue_metrics.ue_metric import UEMetric
from estimators.estimator import Estimator
from stat_calculators.stat_calculator import StatCalculator, STAT_CALCULATORS, STAT_DEPENDENCIES
from stat_calculators import EmbeddingsCalculator


def _order_calculators(stats: List[str]) -> List[StatCalculator]:
    ordered: List[StatCalculator] = []
    have_stats: Set[str] = set()
    while len(stats) > 0:
        stat = stats[0]
        if stat in have_stats:
            stats = stats[1:]
            continue
        dependent = False
        if stat not in STAT_DEPENDENCIES.keys():
            raise Exception(f'Cant find stat calculator for: {stat}')
        for d in STAT_DEPENDENCIES[stat]:
            if d not in have_stats:
                stats = [d] + stats
                if stats.count(d) > 40:
                    raise Exception(f'Found possibly cyclic dependencies: {d}')
                dependent = True
        if not dependent:
            stats = stats[1:]
            ordered.append(STAT_CALCULATORS[stat])
            for new_stat in ordered[-1].stats:
                have_stats.add(new_stat)
    return ordered


def _check_unique_names(xs):
    names = set()
    for x in xs:
        if str(x) in names:
            raise Exception(f'Got multiple __str__ values for {x}')
        names.add(str(x))


def _delete_nans(ue, metric):
    new_ue, new_metric = [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            new_ue.append(ue[i])
            new_metric.append(metric[i])
    return new_ue, new_metric


class UEManager:
    def __init__(
            self,
            data: Dataset,
            model: Model,
            estimators: List[Estimator],
            generation_metrics: List[GenerationMetric],
            ue_metrics: List[UEMetric],
            processors: List[Processor],
            train_data: Dataset = None,
            ignore_exceptions: bool = True,
            ensemble_model: Optional[EnsembleGenerator] = None
    ):
        self.model: Model = model
        self.train_data: Dataset = train_data
        self.ensemble_model = ensemble_model
        self.data: Dataset = data
        self.estimators: List[Estimator] = estimators
        self.generation_metrics: List[GenerationMetric] = generation_metrics
        self.ue_metrics: List[UEMetric] = ue_metrics
        _check_unique_names(generation_metrics)
        _check_unique_names(estimators)
        _check_unique_names(ue_metrics)
        stats = [s for e in estimators for s in e.stats_dependencies] + \
                [s for m in generation_metrics for s in m.stats_dependencies] + ['greedy_tokens', 'greedy_texts']
        self.stat_calculators: List[StatCalculator] = _order_calculators(stats)

        self.gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.metrics: Dict[Tuple[str, str, str, str], float] = {}
        self.stats: Dict[str, List] = defaultdict(list)

        self.processors = processors
        self.ignore_exceptions = ignore_exceptions

    def __call__(self) -> Dict[Tuple[str, str, str, str], float]:
        
        train_embeddings_decoder, train_embeddings_encoder = self.extract_train_embeddings()
        for inp_texts, target_texts in tqdm(self.data):
            target_tokens = [self.model.tokenizer([text])['input_ids'][0] + [self.model.tokenizer.eos_token_id]
                             for text in target_texts]
            batch_stats: Dict[str, np.ndarray] = {}
            for key, val in [
                ('input_texts', inp_texts),
                ('target_texts', target_texts),
                ('target_tokens', target_tokens),
            ]:
                self.stats[key] += val
                batch_stats[key] = val
                
            batch_stats['generation_params'] = {}
            batch_stats['ensemble_model'] = self.ensemble_model
                        
            try:
                for stat_calculator in self.stat_calculators:
                    new_stats = stat_calculator(batch_stats, inp_texts, self.model)
                    for stat, stat_value in new_stats.items():
                        if stat in batch_stats.keys():
                            continue
                        batch_stats[stat] = stat_value
            except Exception as e:
                if self.ignore_exceptions:
                    sys.stderr.write(f'Caught exception while calculating stats: {e}')
                    continue
                else:
                    raise e

            if len(train_embeddings_decoder):
                batch_stats["train_embeddings_decoder"] = train_embeddings_decoder
            if len(train_embeddings_encoder):
                batch_stats["train_embeddings_encoder"] = train_embeddings_encoder
            
            batch_estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for estimator in self.estimators:
                e = estimator(batch_stats).tolist()
                self.estimations[estimator.level, str(estimator)] += e
                batch_estimations[estimator.level, str(estimator)] += e
            batch_gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for generation_metric in self.generation_metrics:
                m = generation_metric(batch_stats, target_texts=target_texts, target_tokens=target_tokens).tolist()
                self.gen_metrics[generation_metric.level, str(generation_metric)] += m
                batch_gen_metrics[generation_metric.level, str(generation_metric)] += m

            for key in ['greedy_texts', 'greedy_tokens']:
                if key in batch_stats.keys():
                    self.stats[key] += batch_stats[key]
            for processor in self.processors:
                processor.on_batch(batch_stats, batch_gen_metrics, batch_estimations)

        for (e_level, e_name), estimator_values in self.estimations.items():
            for (gen_level, gen_name), generation_metric in self.gen_metrics.items():
                for ue_metric in self.ue_metrics:
                    if gen_level != e_level:
                        continue
                    if len(estimator_values) != len(generation_metric):
                        raise Exception(f'Got different number of metrics for {e_name} and {gen_name}: '
                                        f'{len(estimator_values)} and {len(generation_metric)}')
                    ue, metric = _delete_nans(estimator_values, generation_metric)
                    if len(ue) == 0:
                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
                    else:
                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = ue_metric(ue, metric)

        for processor in self.processors:
            processor.on_eval(self.metrics)

        return self.metrics
    
    def extract_train_embeddings(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        train_embeddings_decoder = []
        train_embeddings_encoder = []
        if any([isinstance(stat_calculator, EmbeddingsCalculator) for stat_calculator in self.stat_calculators]) and (self.train_data is not None):
            for stat_calculator in self.stat_calculators:
                if isinstance(stat_calculator, EmbeddingsCalculator):
                    embeddings_calculator = stat_calculator
            for inp_texts, target_texts in tqdm(self.train_data):
                target_tokens = [self.model.tokenizer([text])['input_ids'][0] + [self.model.tokenizer.eos_token_id]
                                 for text in target_texts]

                batch_stats: Dict[str, np.ndarray] = {}
                for key, val in [
                    ('input_texts', inp_texts),
                    ('target_texts', target_texts),
                    ('target_tokens', target_tokens),
                ]:
                    self.stats[key] += val
                    batch_stats[key] = val
                batch_stats = embeddings_calculator(batch_stats, inp_texts, self.model)
                train_embeddings_decoder.append(batch_stats["embeddings_decoder"])
                if "embeddings_encoder" in batch_stats.keys():
                    train_embeddings_encoder.append(batch_stats["embeddings_encoder"])
            train_embeddings_decoder = torch.cat(train_embeddings_decoder)
            if len(train_embeddings_encoder):
                train_embeddings_encoder = torch.cat(train_embeddings_encoder)
            self.stat_calculators.remove(embeddings_calculator)
        return train_embeddings_decoder, train_embeddings_encoder

    def save(self, save_path: str):
        if len(self.metrics) == 0:
            raise Exception('Nothing to save')
        torch.save({
            'metrics': self.metrics,
            'gen_metrics': self.gen_metrics,
            'estimations': self.estimations,
            'stats': self.stats,
        }, save_path)

    @staticmethod
    def load(load_path: str) -> 'UEManager':
        res_dict = torch.load(load_path)
        man = UEManager(None, None, [], [], [], [])
        man.metrics = res_dict.get('metrics', None)
        man.gen_metrics = res_dict.get('gen_metrics', None)
        man.estimations = res_dict.get('estimations', None)
        man.stats = res_dict.get('stats', None)
        return man
