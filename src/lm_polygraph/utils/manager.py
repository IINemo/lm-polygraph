zimport numpy as np
import torch
import sys

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Union
from tqdm import tqdm
from dataclasses import dataclass

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, Model
from lm_polygraph.utils.processor import Processor
from lm_polygraph.utils.normalize import normalize_ue, can_normalize_ue
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.ue_metrics.ue_metric import UEMetric
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator, STAT_CALCULATORS, STAT_DEPENDENCIES
from lm_polygraph.stat_calculators import EmbeddingsCalculator


def _order_calculators(stats: List[str]) -> Tuple[List[str], Set[str]]:
    ordered: List[str] = []
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
            ordered.append(stat)
            for new_stat in STAT_CALCULATORS[stat].stats:
                have_stats.add(new_stat)
    return ordered, have_stats


def _check_unique_names(xs):
    names = set()
    for x in xs:
        if str(x) in names:
            raise Exception(f'Got multiple __str__ values for {x}')
        names.add(str(x))


def _delete_nans(ue, metric):
    new_ue, new_metric, selected_ids = [], [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            if not isinstance(ue[i], complex):
                new_ue.append(ue[i])
            else:
                new_ue.append(ue[i].real)
            new_metric.append(metric[i])
            selected_ids.append(i)
    return new_ue, new_metric, selected_ids


def _recombine_data(ue, gen_metric, inputs):
    ue = np.array(ue)
    gen_metric = np.array(gen_metric)
     
    # np.unique() with return_counts=True?
    recombined_inputs = defaultdict(list)
    for i, input_text in enumerate(inputs):
        recombined_inputs[input_text].append(i)
    
    recombined_ue, recombined_gen_metric = [], []
    for input_text, ids in recombined_inputs.items():
        recombined_ue.append(ue[ids].mean()) 
        # Assumes that metric is bigger for better generations!
        recombined_gen_metric.append(gen_metric[ids].max()) 

    return recombined_ue, recombined_gen_metric


@dataclass
class UncertaintyOutput:
    generation_text: str
    uncertainty: Union[List[float], float]


def estimate_uncertainty(model: WhiteboxModel, estimator: Estimator, input_text: str):
    man = UEManager(Dataset([input_text], [''], batch_size=1), model,
                    [estimator], [], [], [], ignore_exceptions=False, verbose=False)
    man()
    ue = man.estimations[estimator.level, str(estimator)]
    if can_normalize_ue(estimator, model.model_path):
        if estimator.level == 'sequence':
            ue = normalize_ue(estimator, model.model_path, ue[0])
        else:
            ue = [normalize_ue(estimator, model.model_path, i) for i in ue]
    texts = man.stats.get('greedy_texts', man.stats.get('blackbox_greedy_texts', None))
    return UncertaintyOutput(texts[0], ue[0])


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
            background_train_data: Dataset = None,
            ignore_exceptions: bool = True,
            ensemble_model: Optional[WhiteboxModel] = None,
            deberta_batch_size: int = 10,
            verbose: bool = True,
            max_new_tokens: int = 100,
            background_train_dataset_max_new_tokens: int = 100,
            ood_detection: bool = False,
    ):
        self.model: WhiteboxModel = model
        self.train_data: Dataset = train_data
        self.background_train_data: Dataset = background_train_data
        self.ensemble_model = ensemble_model
        self.deberta_batch_size = deberta_batch_size
        self.data: Dataset = data
        self.estimators: List[Estimator] = estimators
        self.generation_metrics: List[GenerationMetric] = generation_metrics
        self.ue_metrics: List[UEMetric] = ue_metrics
        _check_unique_names(generation_metrics)
        _check_unique_names(estimators)
        _check_unique_names(ue_metrics)

        if isinstance(model, BlackboxModel):
            greedy = ['blackbox_greedy_texts']
        else:
            greedy = ['greedy_tokens', 'greedy_texts']
        stats = [s for e in estimators for s in e.stats_dependencies] + \
                [s for m in generation_metrics for s in m.stats_dependencies] + greedy
        stats, have_stats = _order_calculators(stats)
        stats = [s for s in stats if not (str(s).startswith('blackbox_') and s[len('blackbox_'):] in have_stats)]
        self.stat_calculators: List[StatCalculator] = [STAT_CALCULATORS[c] for c in stats]
        if verbose:
            print('Stat calculators:', self.stat_calculators)

        train_stats = [s for e in estimators for s in e.stats_dependencies if s.startswith("train")]
        train_stats += ['greedy_tokens', 'greedy_texts'] if "train_greedy_log_likelihoods" in train_stats else []     
        train_stats, _ = _order_calculators(train_stats)
        self.train_stat_calculators: List[StatCalculator] = [STAT_CALCULATORS[c] for c in train_stats]
        background_train_stats = [s for e in estimators for s in e.stats_dependencies if s.startswith("background_train")]
        background_train_stats, _ = _order_calculators(background_train_stats)
        self.background_train_stat_calculators: List[StatCalculator] = [STAT_CALCULATORS[c] for c in background_train_stats]

        self.gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.metrics: Dict[Tuple[str, str, str, str], float] = {}
        self.stats: Dict[str, List] = defaultdict(list)

        self.ood_detection = ood_detection
        self.processors = processors
        self.ignore_exceptions = ignore_exceptions
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens
        self.background_train_dataset_max_new_tokens = background_train_dataset_max_new_tokens
        
    def __call__(self) -> Dict[Tuple[str, str, str, str], float]:
        train_stats = self.extract_train_embeddings()
        background_train_stats = self.extract_train_embeddings(background=True)

        iterable_data = tqdm(self.data) if self.verbose else self.data
        for inp_texts, target_texts, ood_labels in iterable_data:
            batch_stats: Dict[str, np.ndarray] = {}
            for key, val in [
                ('input_texts', inp_texts),
                ('target_texts', target_texts),
                ('ood_labels', ood_labels),
            ]:
                self.stats[key] += val
                batch_stats[key] = val

            if isinstance(self.model, WhiteboxModel):
                target_tokens = [self.model.tokenizer([text])['input_ids'][0] + [self.model.tokenizer.eos_token_id]
                                 for text in target_texts]
                self.stats['target_tokens'] += target_tokens
                batch_stats['target_tokens'] = target_tokens

            batch_stats['generation_params'] = {}
            batch_stats['ensemble_model'] = self.ensemble_model
            batch_stats['deberta_batch_size'] = self.deberta_batch_size

            try:
                for stat_calculator in self.stat_calculators:
                    new_stats = stat_calculator(batch_stats, inp_texts, self.model, self.max_new_tokens)
                    for stat, stat_value in new_stats.items():
                        if stat in batch_stats.keys():
                            continue
                        batch_stats[stat] = stat_value
                        if f'blackbox_{stat}' in STAT_CALCULATORS.keys():
                            batch_stats[f'blackbox_{stat}'] = stat_value
            except Exception as e:
                if self.ignore_exceptions:
                    log_msg = f'Caught exception while calculating stats: {e} in Stat Calculator {stat_calculator}'
                    sys.stderr.write(log_msg)
                    print(log_msg)
                    continue
                else:
                    raise e

            for stat in train_stats.keys():
                batch_stats[stat] = train_stats[stat]
            for stat in background_train_stats.keys():
                batch_stats[stat] = background_train_stats[stat]

            batch_estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            bad_estimators = []
            
            for estimator in self.estimators:
                try:
                    e = estimator(batch_stats).tolist()
                    self.estimations[estimator.level, str(estimator)] += e
                    batch_estimations[estimator.level, str(estimator)] += e
                except Exception as e:
                    if self.ignore_exceptions:
                        bad_estimators.append(estimator)
                        log_msg = f'Caught exception while estimating uncertainty: {e} in estimator {estimator}. Estimator will be removed.'
                        sys.stderr.write(log_msg)
                        print(log_msg)
                        continue
                    else:
                        raise e
            for bad_estimator in bad_estimators:
                key = (bad_estimator.level, str(bad_estimator))
                self.estimations.pop(key, None)
                self.estimators.remove(bad_estimator)
                
                
            batch_gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for generation_metric in self.generation_metrics:
                m = generation_metric(batch_stats, target_texts=target_texts, target_tokens=target_tokens).tolist()
                self.gen_metrics[generation_metric.level, str(generation_metric)] += m
                batch_gen_metrics[generation_metric.level, str(generation_metric)] += m

            for key in ['blackbox_greedy_texts', 'greedy_texts', 'greedy_tokens']:
                if key in batch_stats.keys():
                    self.stats[key] += batch_stats[key]

            for processor in self.processors:
                processor.on_batch(batch_stats, batch_gen_metrics, batch_estimations)

        print(self.gen_metrics)
        for (e_level, e_name), estimator_values in self.estimations.items():
            for (gen_level, gen_name), generation_metric in self.gen_metrics.items():
                for ue_metric in self.ue_metrics:
                    if gen_level != e_level:
                        continue
                    if len(estimator_values) != len(generation_metric):
                        raise Exception(f'Got different number of metrics for {e_name} and {gen_name}: '
                                        f'{len(estimator_values)} and {len(generation_metric)}')
                        
                    if hasattr(ue_metric, 'is_ood_metric') and ue_metric.is_ood_metric and self.ood_detection:
                        generation_metric = self.stats['ood_labels']
                        gen_name = "ood"
                    
                    if (e_level, e_name, gen_name, str(ue_metric)) in self.metrics.keys():
                        continue
                    # TODO: Report how many nans!
                    # This is important to know for a user
                    ue, metric, selected_ids = _delete_nans(estimator_values, generation_metric)
                    if len(ue) == 0:
                        self.metrics[dict_key] = np.nan
                    else:
                        inputs_no_nans = np.array(self.stats['input_texts'])[selected_ids]
                        rec_ue, rec_metric = _recombine_data(ue, metric,
                                                             inputs_no_nans)

                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = ue_metric(rec_ue, rec_metric)

        for processor in self.processors:
            processor.on_eval(self.metrics)

        return self.metrics
    
    def extract_train_embeddings(self, background: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        train_stats = {}
        result_train_stat = {}
        if background:
            data = self.background_train_data 
            stat_calculators = self.background_train_stat_calculators
            max_new_tokens = self.background_train_dataset_max_new_tokens
        else:
            data = self.train_data 
            stat_calculators = self.train_stat_calculators
            max_new_tokens = self.max_new_tokens
        if len(stat_calculators) and (data is not None):
            for inp_texts, target_texts, _ in tqdm(data):
                target_tokens = [self.model.tokenizer([text])['input_ids'][0] + [self.model.tokenizer.eos_token_id]
                                 for text in target_texts]

                batch_stats: Dict[str, np.ndarray] = {}
                for key, val in [
                    ('input_texts', inp_texts),
                    ('target_texts', target_texts),
                    ('target_tokens', target_tokens),
                ]:
                    batch_stats[key] = val
                    
                for stat_calculator in stat_calculators:
                    new_stats = stat_calculator(batch_stats, inp_texts, self.model, max_new_tokens)
                    for stat, stat_value in new_stats.items():
                        if stat in batch_stats.keys():
                            continue
                        batch_stats[stat] = stat_value
                        
                for stat in batch_stats.keys():
                    if stat in ["input_tokens", "input_texts", "target_texts", "target_tokens"]:
                        continue
                    if stat in train_stats.keys():
                        train_stats[stat].append(batch_stats[stat])
                    else:
                        train_stats[stat] = [batch_stats[stat]]
                            
            key_prefix = "background_train_" if background else "train_"
            for stat in train_stats.keys():
                if isinstance(train_stats[stat][0], list):
                    result_train_stat[key_prefix + stat] = [item for sublist in train_stats[stat] for item in sublist]
                else:
                    result_train_stat[key_prefix + stat] = np.concatenate(train_stats[stat])
                    
        return result_train_stat

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
