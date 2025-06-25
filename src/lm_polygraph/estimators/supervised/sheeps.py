import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from transformers import set_seed, AutoConfig

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from .common import cross_val_hp, TrainerMLP


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Linear(embedding_size, 1)

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float("inf")
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (x * attn_weights).sum(dim=1)


class MLP(nn.Module):
    def __init__(self, n_features: int = 4096):
        super().__init__()
        self.pooling = AttentionPooling(n_features)
        self.output = nn.Linear(n_features, 2)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, mask, eval: bool = False, regression: bool = False):
        x = self.pooling(x, mask)
        x = self.output(x)
        if eval:
            return self.activation(x)[:, 1]
        return x


class LayerSheeps(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        layer: int = -1,
        device: str = "cuda",
        metric_thr: float = 0.3,
    ):
        super().__init__(
            ["token_embeddings", "train_token_embeddings", "train_metrics"], "sequence"
        )
        self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        self.embeddings_type = embeddings_type
        self.device = device
        self.metric_thr = metric_thr
        self.is_fitted = False
        self.params = {
            "n_epochs": [5, 10, 20],
            "batch_size": [32, 64],
            "lr": [1e-1, 1e-2, 1e-3],
            "n_features": [4096],
        }

        self.loss_fn = nn.CrossEntropyLoss()
        self.model_init = lambda param: TrainerMLP(
            n_epochs=param[0],
            batch_size=param[1],
            lr=param[2],
            n_features=param[3],
            device=self.device,
            loss_fn=self.loss_fn,
            model=MLP,
        )

    def __str__(self):
        return f"LayerSheeps_{self.embeddings_type}{self.layer_name}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            train_metrics = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_metrics = (train_metrics < self.metric_thr).astype(int)
            train_embeddings = stats[
                f"train_token_embeddings_{self.embeddings_type}{self.layer_name}"
            ]
            k = 0
            aggregated_embeddings, lens = [], []
            for tokens in train_greedy_tokens:
                aggregated_embeddings.append(
                    torch.tensor(np.array(train_embeddings[k : k + len(tokens)]))
                )
                lens.append(len(tokens))
                k += len(tokens)
            aggregated_embeddings = pad_sequence(
                aggregated_embeddings, batch_first=True, padding_value=0
            )
            attention_mask = np.zeros(
                (aggregated_embeddings.shape[0], aggregated_embeddings.shape[1])
            )

            for i, l in enumerate(lens):
                attention_mask[i, l:] = 1
            attention_mask = torch.tensor(attention_mask).int()

            self.params["n_features"] = [aggregated_embeddings.shape[-1]]
            best_params = cross_val_hp(
                aggregated_embeddings,
                train_metrics,
                self.model_init,
                self.params,
                regression=False,
                mask=attention_mask,
                estimator_name=self.__str__(),
            )
            self.ue_predictor = self.model_init(best_params)

            self.ue_predictor.fit(aggregated_embeddings, train_metrics, attention_mask)
            self.is_fitted = True
        # Inference
        embeddings = stats[f"token_embeddings_{self.embeddings_type}{self.layer_name}"]
        greedy_tokens = stats["greedy_tokens"]
        k = 0
        aggregated_embeddings, lens = [], []
        for tokens in greedy_tokens:
            aggregated_embeddings.append(
                torch.tensor(np.array(embeddings[k : k + len(tokens)]))
            )
            lens.append(len(tokens))
            k += len(tokens)
        aggregated_embeddings = pad_sequence(
            aggregated_embeddings, batch_first=True, padding_value=0
        )
        attention_mask = np.zeros(
            (aggregated_embeddings.shape[0], aggregated_embeddings.shape[1])
        )
        for i, l in enumerate(lens):
            attention_mask[i, l:] = 1

        attention_mask = torch.tensor(attention_mask).int()
        ue = self.ue_predictor.predict(aggregated_embeddings, attention_mask)

        return ue


class Sheeps(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        layers: List[int] = None,
        device: str = "cuda",
        metric_thr: float = 0.3,
        dev_size: float = 0.5,
        model_name: str = None,
    ):
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        if layers is None:
            self.layers = (
                list(range(self.model_config.num_hidden_layers))
                if hasattr(self.model_config, "num_hidden_layers")
                else list(range(self.model_config.text_config.num_hidden_layers))
            )
        else:
            self.layers = layers
        self.embeddings_type = embeddings_type
        self.device = device
        self.metric_thr = metric_thr
        self.dev_size = dev_size
        self.is_fitted = False
        self.layersheeps = [
            LayerSheeps(
                embeddings_type=embeddings_type,
                layer=layer,
                device=device,
                metric_thr=metric_thr,
            )
            for layer in self.layers
        ]

        super().__init__(
            ["token_embeddings", "train_token_embeddings", "train_metrics"], "sequence"
        )
        self.ue_predictor = LogisticRegressionCV()

    def __str__(self):
        return f"Sheeps_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            set_seed(42)
            train_metrics_raw = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_metrics = (train_metrics_raw < self.metric_thr).astype(int)
            train_idx, dev_idx = train_test_split(
                np.arange(len(train_greedy_tokens)),
                test_size=self.dev_size,
                random_state=42,
            )
            train_sheeps = []
            for layer in self.layers:
                # Prepare stats for this layer
                layer_name = "" if layer == -1 else f"_{layer}"
                train_embeddings = stats[
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}"
                ]
                k = 0
                aggregated_embeddings = []
                for tokens in train_greedy_tokens:
                    aggregated_embeddings.append(train_embeddings[k : k + len(tokens)])
                    k += len(tokens)
                train_stats = {
                    "train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx],
                    "greedy_tokens": [train_greedy_tokens[k] for k in dev_idx],
                    "train_metrics": train_metrics_raw[train_idx],
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}": [
                        emb for k in train_idx for emb in aggregated_embeddings[k]
                    ],
                    f"token_embeddings_{self.embeddings_type}{layer_name}": [
                        emb for k in dev_idx for emb in aggregated_embeddings[k]
                    ],
                }
                score = self.layersheeps[layer](train_stats).reshape(-1)
                train_sheeps.append(score)
            train_sheeps = np.array(train_sheeps).T
            self.ue_predictor.fit(train_sheeps, train_metrics[dev_idx])
            self.is_fitted = True

        eval_scores = []
        for layer in self.layers:
            score = self.layersheeps[layer](stats).reshape(-1)
            eval_scores.append(score)
        eval_scores = np.array(eval_scores).T
        eval_scores[np.isnan(eval_scores)] = 0
        ue = self.ue_predictor.predict_proba(eval_scores)[:, 1]
        return ue
