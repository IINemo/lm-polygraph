import numpy as np
import torch.nn as nn

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator

from lm_polygraph.estimators.mahalanobis_distance import (
    create_cuda_tensor_from_numpy,
)
from transformers import set_seed, AutoConfig
from .common import cross_val_hp, TrainerMLP

import logging

log = logging.getLogger("lm_polygraph")


class MLP(nn.Module):
    def __init__(self, n_features: int = 4096):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.indentity_activation = nn.Identity()
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, x, eval: bool = False, regression: bool = False):
        x = self.model(x)
        if regression:
            return self.indentity_activation(x)
        return self.sigmoid_activation(x)


class SAPLMA(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        layer: int = None,
        device: str = "cuda",
        model_name: str = None,
    ):
        self.layer = layer
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        if layer is None:
            self.layer = (
                self.model_config.num_hidden_layers // 2
                if hasattr(self.model_config, "num_hidden_layers")
                else self.model_config.text_config.num_hidden_layers // 2
            )
        else:
            self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        self.embeddings_type = embeddings_type
        self.device = device
        self.is_fitted = False

        super().__init__(
            [
                "train_embeddings",
                "embeddings",
                "train_metrics",
            ],
            "sequence",
        )

        # MLP hyperparameters
        self.params = {
            "n_epochs": [5, 10],
            "batch_size": [64, 128],
            "lr": [1e-3, 1e-4, 5e-5, 1e-5, 5e-6],
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
        return f"SAPLMA_{self.embeddings_type}{self.layer_name}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Select embeddings
        emb_key = f"embeddings_{self.embeddings_type}{self.layer_name}"
        embeddings = create_cuda_tensor_from_numpy(stats[emb_key])

        # Fit model if not already fitted
        if not self.is_fitted:
            set_seed(42)
            target = np.array(stats["train_metrics"])

            # Check if target is continuous or in [0, 1]
            unique_vals = np.unique(target)
            if np.all((unique_vals == 0) | (unique_vals == 1)):
                self.regression = False
                self.loss_fn = nn.BCELoss()
            else:
                self.regression = True
                self.loss_fn = nn.MSELoss()

            train_embeddings_key = (
                f"train_embeddings_{self.embeddings_type}{self.layer_name}"
            )
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[train_embeddings_key]
            )

            # Hyperparameter search
            self.params["n_features"] = [train_embeddings.shape[-1]]
            best_params = cross_val_hp(
                train_embeddings,
                1 - target,
                self.model_init,
                self.params,
                regression=self.regression,
                estimator_name=self.__str__(),
            )
            self.ue_predictor = self.model_init(best_params)

            self.ue_predictor.fit(
                train_embeddings, 1 - target, regression=self.regression
            )
            self.is_fitted = True

        # Predict
        ue = self.ue_predictor.predict(embeddings)

        return ue
