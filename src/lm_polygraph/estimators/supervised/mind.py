import numpy as np
from typing import Dict
import torch
import torch.nn as nn

from lm_polygraph.estimators.estimator import Estimator
from .common import cross_val_hp, TrainerMLP


class MLP(nn.Module):
    def __init__(self, n_features: int = 4096):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, regression: bool = False, eval: bool = False):
        x = self.model(x)
        if eval:
            return self.softmax(x)[:, 1]
        return x


def aggregate_token_embeddings(embeddings):
    return (np.mean(embeddings, axis=0) + np.array(embeddings[-1])) / 2


class MIND(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        layer: int = -1,
        device: str = "cuda",
        metric_thr: float = 0.3,
    ):
        super().__init__(
            [
                "token_embeddings",
                "train_token_embeddings",
                "train_metrics",
            ],
            "sequence",
        )
        self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        self.embeddings_type = embeddings_type

        self.device = device
        self.metric_thr = metric_thr
        self.is_fitted = False

        # MLP hyperparameters
        self.params = {
            "n_epochs": [10, 20],
            "batch_size": [32, 64],
            "lr": [1e-3, 1e-4, 5e-4, 5e-5, 1e-5, 5e-6],
            "n_features": [4096],
        }
        self.loss_fn = None
        self.model_init = lambda param: TrainerMLP(
            n_epochs=param[0],
            batch_size=param[1],
            lr=param[2],
            n_features=param[3],
            device=self.device,
            loss_fn=self.loss_fn,
            model=MLP,
        )
        self.ue_predictor = TrainerMLP(device=self.device, model=MLP)

    def __str__(self):
        return f"MIND_{self.embeddings_type}{self.layer_name}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:

            train_metrics = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_metrics = (train_metrics < self.metric_thr).astype(int)

            nSamples = [(train_metrics == 0).sum(), (train_metrics == 1).sum()]
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            normedWeights = torch.FloatTensor(normedWeights).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=normedWeights).to(self.device)

            train_embeddings = stats[
                f"train_token_embeddings_{self.embeddings_type}{self.layer_name}"
            ]

            k = 0
            aggregated_embeddings = []
            for tokens in train_greedy_tokens:
                aggregated_embeddings.append(
                    aggregate_token_embeddings(train_embeddings[k : k + len(tokens)])
                )
                k += len(tokens)
            aggregated_embeddings = np.array(aggregated_embeddings)

            self.params["n_features"] = [aggregated_embeddings.shape[-1]]
            best_params = cross_val_hp(
                aggregated_embeddings,
                train_metrics,
                self.model_init,
                self.params,
                regression=False,
                estimator_name=self.__str__(),
            )
            self.ue_predictor = self.model_init(best_params)
            self.ue_predictor.fit(
                aggregated_embeddings, train_metrics, regression=False
            )
            self.is_fitted = True

        k = 0
        embeddings = stats[f"token_embeddings_{self.embeddings_type}{self.layer_name}"]
        greedy_tokens = stats["greedy_tokens"]
        aggregated_embeddings = []
        for tokens in greedy_tokens:
            aggregated_embeddings.append(
                aggregate_token_embeddings(embeddings[k : k + len(tokens)])
            )
            k += len(tokens)
        aggregated_embeddings = np.array(aggregated_embeddings)
        ue = self.ue_predictor.predict(aggregated_embeddings)
        return ue
