import gc
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel
from .greedy_visual_probs import GreedyProbsVisualCalculator


class TrainingStatisticExtractionCalculatorVisual(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "train_embeddings",
            "background_train_embeddings",
            "train_greedy_log_likelihoods",
        ], []

    def __init__(self, train_dataset=None, background_train_dataset=None):
        super().__init__()
        self.hidden_layer = -1
        self.train_dataset = train_dataset
        self.background_train_dataset = background_train_dataset
        self.statistics_extracted = False
        self.base_calculators = [GreedyProbsVisualCalculator()]

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: VisualWhiteboxModel,
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

            for dataset, dataset_name in zip(datasets, datasets_name):
                if dataset is None:
                    continue

                train_max_new_tokens = (
                    max_new_tokens
                    if dataset_name == "train_"
                    else background_train_dataset_max_new_tokens
                )

                for inp_texts, target_texts, images in tqdm(dataset):
                    batch_stats: Dict[str, np.ndarray] = {
                        "images": model.get_images(images),
                        "input_texts": inp_texts,
                        "target_texts": target_texts,
                    }

                    for stat_calculator in self.base_calculators:
                        new_stats = stat_calculator(
                            batch_stats, inp_texts, model, train_max_new_tokens
                        )
                        for stat, stat_value in new_stats.items():
                            if stat in batch_stats.keys():
                                continue
                            batch_stats[stat] = stat_value

                    for stat in batch_stats.keys():
                        if stat in [
                            "input_tokens",
                            "input_texts",
                            "target_texts",
                            "images",
                        ]:
                            continue
                        key = dataset_name + stat
                        if key in train_stats:
                            train_stats[key].append(batch_stats[stat])
                        else:
                            train_stats[key] = [batch_stats[stat]]

                    torch.cuda.empty_cache()
                    gc.collect()

            for stat in train_stats.keys():
                if (
                    any(s is None for s in train_stats[stat])
                    or ("tokenizer" in stat)
                    or ("processor" in stat)
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
