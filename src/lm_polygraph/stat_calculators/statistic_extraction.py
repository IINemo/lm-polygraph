import gc
import torch
import numpy as np
from tqdm import tqdm

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from .greedy_probs import GreedyProbsCalculator
from .embeddings import EmbeddingsCalculator, TokenEmbeddingsCalculator
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from .attention import LookbackRatioCalculator, AttentionFeaturesCalculator


class TrainingStatisticExtractionCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "train_embeddings",
            "train_token_embeddings",
            "background_train_embeddings",
            "background_train_token_embeddings",
            "train_greedy_log_likelihoods",
            "train_lookback_ratios",
            "train_attention_features",
            "train_metrics",
        ], []

    def __init__(
        self,
        train_dataset=None,
        background_train_dataset=None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        return_embeddings: bool = False,
        return_token_embeddings: bool = False,
        return_lookback_ratios: bool = False,
        return_attention_features: bool = False,
        target_metric: GenerationMetric = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.background_train_dataset = background_train_dataset
        self.statistics_extracted = False
        self.base_calculators = [
            GreedyProbsCalculator(
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        ]
        if return_embeddings:
            self.base_calculators.append(EmbeddingsCalculator())
        if return_token_embeddings:
            self.base_calculators.append(TokenEmbeddingsCalculator())
        if return_lookback_ratios:
            self.base_calculators.append(LookbackRatioCalculator())
        if return_attention_features:
            self.base_calculators.append(AttentionFeaturesCalculator())
        if target_metric is not None:
            self.target_metric = target_metric

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        background_train_dataset_max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        if self.statistics_extracted:
            return {}
        else:
            train_stats = {}
            result_train_stat = {}
            datasets = [self.train_dataset, self.background_train_dataset]
            datasets_name = ["train_", "background_train_"]
            skip_keywords = ["tokenizer", "layers", "_raw"]
            for dataset, dataset_name in zip(datasets, datasets_name):
                if dataset is None:
                    continue
                train_max_new_tokens = (
                    max_new_tokens
                    if datasets_name == "train_"
                    else background_train_dataset_max_new_tokens
                )
                for inp_texts, target_texts in tqdm(dataset):
                    batch_stats: Dict[str, np.ndarray] = {}
                    for key, val in [
                        ("input_texts", inp_texts),
                        ("target_texts", target_texts),
                    ]:
                        batch_stats[key] = val
                        batch_stats["layers"] = dependencies["layers"]

                    for stat_calculator in self.base_calculators:
                        new_stats = stat_calculator(
                            batch_stats, inp_texts, model, train_max_new_tokens
                        )
                        for stat, stat_value in new_stats.items():
                            if stat in batch_stats.keys():
                                continue
                            batch_stats[stat] = stat_value

                    if dataset_name == "train_":
                        batch_stats["metrics"] = self.target_metric(
                            batch_stats, target_texts
                        )

                    for stat in batch_stats.keys():
                        if stat in [
                            "input_tokens",
                            "input_texts",
                            "target_texts",
                        ] or any(keyword in stat for keyword in skip_keywords):
                            continue
                        if dataset_name + stat in train_stats.keys():
                            train_stats[dataset_name + stat].append(batch_stats[stat])
                        else:
                            train_stats[dataset_name + stat] = [batch_stats[stat]]

                    torch.cuda.empty_cache()
                    gc.collect()

            for stat in train_stats.keys():
                if any(s is None for s in train_stats[stat]) or any(
                    keyword in stat for keyword in skip_keywords
                ):
                    continue
                if isinstance(train_stats[stat][0], list):
                    result_train_stat[stat] = [
                        item for sublist in train_stats[stat] for item in sublist
                    ]
                else:
                    result_train_stat[stat] = np.concatenate(train_stats[stat])
            self.statistics_extracted = True

            return result_train_stat
