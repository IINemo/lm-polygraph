import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator

from lm_polygraph.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph.stat_calculators.entropy import EntropyCalculator

from .token_mahalanobis_distance import TokenMahalanobisDistance
from .relative_token_mahalanobis_distance import RelativeTokenMahalanobisDistance

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoConfig


class SATRMD(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        metric_thr: float = 0.3,
        aggregation: str = "mean",
        layers: List[int] = None,
        base_method: str = "RelativeTokenMahalanobis",
        device: str = "cuda",
        storage_device: str = "cuda",
        n_pca_components: int = 10,
        dev_size: float = 0.5,
        model_name: str = None,
    ):
        super().__init__(
            [
                "token_embeddings",
                "train_token_embeddings",
                "background_train_token_embeddings",
                "train_metrics",
            ],
            "sequence",
        )
        self.base_method = base_method
        self.embeddings_type = embeddings_type
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
        self.device = device
        self.storage_device = storage_device
        self.n_pca_components = n_pca_components
        self.dev_size = dev_size
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.msp = MaximumSequenceProbability()
        self.ent = EntropyCalculator()
        self.uq_predictor = Ridge()

        self.tmds = {}
        for layer in self.layers:
            if self.base_method == "TokenMahalanobis":
                self.tmds[layer] = TokenMahalanobisDistance(
                    embeddings_type,
                    metric_thr=metric_thr,
                    aggregation="none",
                    layer=layer,
                    device=self.device,
                    storage_device=self.storage_device,
                )
            elif self.base_method == "RelativeTokenMahalanobis":
                self.tmds[layer] = RelativeTokenMahalanobisDistance(
                    embeddings_type,
                    metric_thr=metric_thr,
                    aggregation="none",
                    layer=layer,
                    device=self.device,
                    storage_device=self.storage_device,
                )
            else:
                raise ValueError(f"Invalid base method: {self.base_method}")

    def __str__(self):
        MD = "RMD" if "Relative" in self.base_method else "MD"
        return f"SAT{MD}_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:

        if not self.is_fitted:
            train_metrics = stats["train_metrics"]
            train_greedy_texts = stats["train_greedy_texts"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_mds = []
            train_idx, dev_idx = train_test_split(
                list(range(len(train_metrics))),
                test_size=self.dev_size,
                shuffle=True,
                random_state=42,
            )
            lens = np.array([0] + [len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate(
                [np.arange(tokens_before[i], tokens_before[i + 1]) for i in train_idx]
            )
            token_dev_idx = np.concatenate(
                [np.arange(tokens_before[i], tokens_before[i + 1]) for i in dev_idx]
            )

            for layer in tqdm(
                self.layers,
                desc=f"Fitting layer-wise Mahalanobis distances for {self.__str__()}",
            ):
                layer_name = "" if layer == -1 else f"_{layer}"
                train_token_embeddings = stats[
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}"
                ]

                train_stats = {
                    "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                    "train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx],
                    "greedy_tokens": [train_greedy_tokens[k] for k in dev_idx],
                    "train_metrics": [train_metrics[k] for k in train_idx],
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}": [
                        train_token_embeddings[k] for k in token_train_idx
                    ],
                    f"token_embeddings_{self.embeddings_type}{layer_name}": [
                        train_token_embeddings[k] for k in token_dev_idx
                    ],
                    f"background_train_token_embeddings_{self.embeddings_type}{layer_name}": stats[
                        f"background_train_token_embeddings_{self.embeddings_type}{layer_name}"
                    ],
                }

                centroid_key = (
                    f"centroid{layer_name}_{self.metric_thr}_{len(train_idx)}"
                )
                covariance_key = (
                    f"covariance{layer_name}_{self.metric_thr}_{len(train_idx)}"
                )

                background_centroid_key = f"background_centroid{layer_name}_{self.metric_thr}_{len(train_idx)}"
                background_covariance_key = f"background_covariance{layer_name}_{self.metric_thr}_{len(train_idx)}"

                if centroid_key in stats.keys():
                    train_stats[centroid_key] = stats[centroid_key]
                if covariance_key in stats.keys():
                    train_stats[covariance_key] = stats[covariance_key]
                if background_centroid_key in stats.keys():
                    train_stats[background_centroid_key] = stats[
                        background_centroid_key
                    ]
                if background_covariance_key in stats.keys():
                    train_stats[background_covariance_key] = stats[
                        background_covariance_key
                    ]

                md = self.tmds[layer](train_stats, save_data=False).reshape(-1)

                if "Relative" in self.base_method:
                    if background_centroid_key not in stats.keys():
                        stats[background_centroid_key] = self.tmds[layer].centroid_0
                    if background_covariance_key not in stats.keys():
                        stats[background_covariance_key] = self.tmds[layer].sigma_inv_0
                    if centroid_key not in stats.keys():
                        stats[centroid_key] = self.tmds[layer].MD.centroid
                    if covariance_key not in stats.keys():
                        stats[covariance_key] = self.tmds[layer].MD.sigma_inv
                else:
                    if centroid_key not in stats.keys():
                        stats[centroid_key] = self.tmds[layer].centroid
                    if covariance_key not in stats.keys():
                        stats[covariance_key] = self.tmds[layer].sigma_inv

                self.tmds[layer].is_fitted = False
                k = 0
                mean_md = []
                for tokens in [train_greedy_tokens[k] for k in dev_idx]:
                    dists_i = md[k : k + len(tokens)]
                    k += len(tokens)
                    mean_md.append(np.mean(dists_i))
                train_mds.append(mean_md)
            X = np.array(train_mds).T
            X[np.isnan(X)] = 0
            self.pca = PCA(n_components=self.n_pca_components)
            X = self.pca.fit_transform(X)

            train_greedy_log_likelihoods = stats["train_greedy_log_likelihoods"]
            train_greedy_log_probs = stats["train_greedy_log_probs"]
            msp = np.array(
                self.msp(
                    {
                        "greedy_log_likelihoods": [
                            train_greedy_log_likelihoods[i] for i in dev_idx
                        ]
                    }
                )
            )
            ent = np.array(
                [
                    np.mean(x)
                    for x in self.ent(
                        {
                            "greedy_log_probs": [
                                train_greedy_log_probs[i] for i in dev_idx
                            ]
                        }
                    )["entropy"]
                ]
            )
            X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])
            y = 1 - train_metrics[dev_idx]
            self.uq_predictor.fit(X, y)
            self.is_fitted = True

        eval_mds = []
        greedy_tokens = stats["greedy_tokens"]
        for layer in self.tmds.keys():
            md = self.tmds[layer](stats).reshape(-1)
            k = 0
            mean_md = []
            for tokens in greedy_tokens:
                dists_i = md[k : k + len(tokens)]
                k += len(tokens)
                mean_md.append(np.mean(dists_i))
            eval_mds.append(mean_md)
        eval_dists = np.array(eval_mds).T
        eval_dists[np.isnan(eval_dists)] = 0
        eval_dists = self.pca.transform(eval_dists)

        msp = np.array(
            self.msp({"greedy_log_likelihoods": stats["greedy_log_likelihoods"]})
        )
        ent = np.array(
            [
                np.mean(x)
                for x in self.ent({"greedy_log_probs": stats["greedy_log_probs"]})[
                    "entropy"
                ]
            ]
        )
        X_eval = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1)])
        ues = self.uq_predictor.predict(X_eval)

        return ues
