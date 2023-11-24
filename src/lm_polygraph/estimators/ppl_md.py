import os
import numpy as np
import copy

from typing import Dict

from .estimator import Estimator
from .mahalanobis_distance import (
    MahalanobisDistanceSeq,
)
from .relative_mahalanobis_distance import RelativeMahalanobisDistanceSeq
from .perplexity import Perplexity
from sklearn.model_selection import train_test_split


def save_array(array, filename):
    with open(filename, "wb") as f:
        np.save(f, array)


def load_array(filename):
    with open(filename, "rb") as f:
        array = np.load(f)
    return array


def rank(target_array, source_array):
    ranks_array = np.array([(x >= source_array).sum() for x in target_array]) / len(
        source_array
    )
    return ranks_array


class PPLMDSeq(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        md_type: str = "MD",
        parameters_path: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            [
                "train_greedy_log_likelihoods",
                "greedy_log_likelihoods",
                "embeddings",
                "train_embeddings",
                "background_train_embeddings",
            ],
            "sequence",
        )
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.md_type = md_type

        self.ppls = []
        self.mds = []

        self.train_ppl = None
        self.train_md = None
        self.is_fitted = False

        self.PPL = Perplexity()
        if self.md_type == "MD":
            self.MD = MahalanobisDistanceSeq(
                embeddings_type, parameters_path, normalize=False
            )
            self.MD_val = MahalanobisDistanceSeq(
                embeddings_type, parameters_path, normalize=False
            )
        elif self.md_type == "RMD":
            self.MD = RelativeMahalanobisDistanceSeq(
                embeddings_type, parameters_path, normalize=False
            )
            self.MD_val = RelativeMahalanobisDistanceSeq(
                embeddings_type, parameters_path, normalize=False
            )
        else:
            raise NotImplementedError

        if self.parameters_path is not None:
            self.full_path = (
                f"{self.parameters_path}/ppl_{self.md_type}_{self.embeddings_type}"
            )
            os.makedirs(self.full_path, exist_ok=True)

            if os.path.exists(f"{self.full_path}/train_md.npy"):
                self.train_ppl = load_array(f"{self.full_path}/train_ppl.npy")
                self.train_md = load_array(f"{self.full_path}/train_md.npy")
                self.is_fitted = True

    def __str__(self):
        return f"PPL{self.md_type}Seq_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ppl = self.PPL(stats)
        md = self.MD(stats)

        if not self.is_fitted:
            copy_stats = copy.deepcopy(stats)
            copy_stats["greedy_log_likelihoods"] = copy_stats[
                "train_greedy_log_likelihoods"
            ]
            self.train_ppl = self.PPL(copy_stats)
            if self.parameters_path is not None:
                save_array(self.train_ppl, f"{self.full_path}/train_ppl.npy")
        if not self.is_fitted:
            train_embeds, val_embeds = train_test_split(
                stats[f"train_embeddings_{self.embeddings_type}"],
                test_size=0.3,
                random_state=42,
            )
            copy_stats = copy.deepcopy(stats)
            copy_stats[f"train_embeddings_{self.embeddings_type}"] = train_embeds
            copy_stats[f"embeddings_{self.embeddings_type}"] = val_embeds
            self.train_md = self.MD_val(copy_stats)
            if self.parameters_path is not None:
                save_array(self.train_md, f"{self.full_path}/train_md.npy")
            self.is_fitted = True

        ppl_rank = rank(ppl, self.train_ppl)
        md_rank = rank(md, self.train_md)

        return (ppl_rank + md_rank) / 2
