import logging
import numpy as np
import itertools
from sklearn.model_selection import KFold
from typing import Dict
from tqdm import tqdm
from sklearn.linear_model import Ridge
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.ue_metrics import PredictionRejectionArea

log = logging.getLogger("lm_polygraph")


def construct_tad_features(
    curr_token_prob,
    token_idx,
    tad_probs,
    attention_features,
    attn_ptr,
    greedy_log_likelihoods,
    seq_idx,
    n_previous_tokens,
    n_attn_features,
):
    """
    Construct TAD features for a given token.
    """
    features = [curr_token_prob]
    for n_prev in range(1, n_previous_tokens + 1):
        if n_prev > token_idx:
            features.extend([0] * (n_attn_features + 3))
        else:
            prev_token_idx = token_idx - n_prev
            prev_tad_prob = np.clip(tad_probs[prev_token_idx], 0, 1)
            features.extend(
                attention_features[attn_ptr][n_prev - 1].tolist()
                + [
                    np.exp(greedy_log_likelihoods[seq_idx][prev_token_idx]),
                    prev_tad_prob,
                    curr_token_prob * prev_tad_prob,
                ]
            )
    return features


def get_tad_ue(
    model,
    greedy_log_likelihoods,
    attention_features,
    val_indices,
    val_token_indices,
    aggregation="mean",
    n_previous_tokens=1,
):
    """
    Compute TAD uncertainty estimates for a validation set.
    """
    tad_scores = []
    attn_ptr = 0
    n_attn_features = attention_features[0].shape[-1]
    for seq_idx in val_indices:
        tad_probs = [np.exp(greedy_log_likelihoods[seq_idx][0])]
        for token_idx in range(1, len(greedy_log_likelihoods[seq_idx])):
            curr_token_prob = np.exp(greedy_log_likelihoods[seq_idx][token_idx])
            features = construct_tad_features(
                curr_token_prob,
                token_idx,
                tad_probs,
                attention_features,
                val_token_indices[attn_ptr],
                greedy_log_likelihoods,
                seq_idx,
                n_previous_tokens,
                n_attn_features,
            )
            pred_prob = model.predict([features])[0]
            pred_prob = np.clip(pred_prob, 0, 1)
            tad_probs.append(pred_prob)
            attn_ptr += 1
        tad_probs = np.array(tad_probs)
        if aggregation == "mean":
            tad_scores.append(-tad_probs.mean())
        elif aggregation == "sum(log(p_i))":
            tad_scores.append(-np.log(tad_probs + 1e-5).sum())
        else:
            raise ValueError(f"Aggregation {aggregation} not supported")
    return tad_scores


def cross_val_hp(
    X,
    y,
    model_init,
    params,
    attention_features,
    greedy_log_likelihoods,
    metrics,
    n_previous_tokens,
    step=1,
):
    """
    Cross-validate hyperparameters for TAD using PRR metric.
    """
    prr_metric = PredictionRejectionArea()
    best_prr = -np.inf
    best_params = None

    param_grid = list(itertools.product(*params.values()))
    for param in tqdm(param_grid, desc=f"Hyperparameter search for TAD (step {step})"):
        model = model_init(param)
        aggregation = param[1]
        prr_scores = []
        lens = np.array([0] + [len(ll) - 1 for ll in greedy_log_likelihoods])
        tokens_before = np.cumsum(lens)
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        for train_indices, val_indices in kf.split(range(len(greedy_log_likelihoods))):
            train_token_indices = np.concatenate(
                [
                    np.arange(tokens_before[i], tokens_before[i + 1])
                    for i in train_indices
                ]
            )
            val_token_indices = np.concatenate(
                [np.arange(tokens_before[i], tokens_before[i + 1]) for i in val_indices]
            )
            X_train = X[train_token_indices]
            y_train = y[train_token_indices]
            model.fit(X_train, y_train)
            tad_scores = get_tad_ue(
                model,
                greedy_log_likelihoods,
                attention_features,
                val_indices,
                val_token_indices,
                aggregation,
                n_previous_tokens,
            )
            metrics_scores = np.array([metrics[i] for i in val_indices])
            prr_scores.append(prr_metric(np.array(tad_scores), metrics_scores))
        prr_mean = np.mean(prr_scores)
        if prr_mean > best_prr:
            best_prr = prr_mean
            best_params = param
    if best_params is None:
        best_params = param_grid[0]
    log.info(
        f"TAD (step {step}) best hyperparameters: {best_params}, best PRR score: {best_prr}"
    )
    return best_params


class TAD(Estimator):
    """
    Implements the TAD (Trainable Attention Dependency) uncertainty estimator from
    (https://arxiv.org/pdf/2408.10692).

    The TAD method uses a two-stage regression approach to estimate token-level uncertainty
    by leveraging attention features and log-likelihoods of generated tokens. In the first
    stage, a regression model is trained to predict a token's correctness using features
    derived from the current and previous tokens' attention and log-likelihood statistics.
    In the second stage, stacking is used: predictions from the first-stage model are used
    as additional features for a final regression model, improving the uncertainty estimation.

    For each sequence, token-level uncertainty scores are aggregated (mean or sum of log)
    to produce a sequence-level uncertainty score.

    Args:
        n_previous_tokens (int): Number of previous tokens to use for feature construction.

    Dependencies:
        - "attention_features"
        - "greedy_log_likelihoods"
        - "train_attention_features"
        - "train_metrics"
    """

    def __init__(self, n_previous_tokens: int = 10):
        super().__init__(
            [
                "attention_features",
                "greedy_log_likelihoods",
                "train_attention_features",
                "train_metrics",
            ],
            "sequence",
        )
        self.n_previous_tokens = n_previous_tokens
        self.model_init = lambda param: Ridge(alpha=param[0])
        self.params = {
            "alpha": [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4],
            "aggregation": ["mean", "sum(log(p_i))"],
        }
        self.is_fitted = False

    def __str__(self):
        return "TAD"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute TAD uncertainty scores for a batch of samples.

        Args:
            stats (Dict[str, np.ndarray]): Dictionary containing attention features,
                greedy log-likelihoods, and train metrics.

        Returns:
            np.ndarray: Array of sequence-level uncertainty scores (lower = more certain).
        """
        # Prepare attention features for test set
        test_attention_features = [
            np.array(item)
            for sublist in stats["attention_features"]
            for item in sublist
        ]
        test_greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        if not self.is_fitted:
            # Prepare training data
            train_attention_features = [
                np.array(item)
                for sublist in stats["train_attention_features"]
                for item in sublist
            ]
            self.n_attn = train_attention_features[0].shape[-1]
            train_greedy_log_likelihoods = stats["train_greedy_log_likelihoods"]
            train_metrics = stats["train_metrics"]

            # First step: extract features from training data
            X, y = [], []
            attn_ptr = 0
            n_sequences = len(train_metrics)
            for seq_idx in range(n_sequences):
                seq_score = np.clip(train_metrics[seq_idx], 0, 1)
                tad_probs = [seq_score]
                X_seq, y_seq = [], []
                for token_idx in range(1, len(train_greedy_log_likelihoods[seq_idx])):
                    curr_token_score = seq_score
                    curr_token_prob = np.exp(
                        train_greedy_log_likelihoods[seq_idx][token_idx]
                    )
                    tad_probs.append(curr_token_score)
                    y_seq.append(curr_token_score)
                    features = construct_tad_features(
                        curr_token_prob,
                        token_idx,
                        tad_probs,
                        train_attention_features,
                        attn_ptr,
                        train_greedy_log_likelihoods,
                        seq_idx,
                        self.n_previous_tokens,
                        self.n_attn,
                    )
                    X_seq.append(features)
                    attn_ptr += 1
                if X_seq:
                    X.append(X_seq)
                    y.append(y_seq)
            X = np.concatenate(X)
            y = np.concatenate(y)

            # First step: cross-validation for initial model
            best_params = cross_val_hp(
                X,
                y,
                self.model_init,
                self.params,
                train_attention_features,
                train_greedy_log_likelihoods,
                train_metrics,
                n_previous_tokens=self.n_previous_tokens,
                step=1,
            )
            self.regression_model = self.model_init(best_params)
            self.regression_model.fit(X, y)

            # Prepare stacking models for second step
            lens = np.array([0] + [len(ll) - 1 for ll in train_greedy_log_likelihoods])
            tokens_before = np.cumsum(lens)
            self.stacking_models = []
            self.fold_idx = {}
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            for fold_id, (train_indices, val_indices) in enumerate(
                kf.split(range(len(train_greedy_log_likelihoods)))
            ):
                fold_model = self.model_init(best_params)
                train_token_indices = np.concatenate(
                    [
                        np.arange(tokens_before[j], tokens_before[j + 1])
                        for j in train_indices
                    ]
                )
                X_train = X[train_token_indices]
                y_train = y[train_token_indices]
                fold_model.fit(X_train, y_train)
                for seq_idx in val_indices:
                    self.fold_idx[seq_idx] = fold_id
                self.stacking_models.append(fold_model)

            # Second step: stacking predictions as features
            X_stack, y_stack = [], []
            attn_ptr = 0
            for seq_idx in range(n_sequences):
                seq_score = np.clip(train_metrics[seq_idx], 0, 1)
                tad_probs = [np.exp(train_greedy_log_likelihoods[seq_idx][0])]
                X_seq, y_seq = [], []
                for token_idx in range(1, len(train_greedy_log_likelihoods[seq_idx])):
                    curr_token_prob = np.exp(
                        train_greedy_log_likelihoods[seq_idx][token_idx]
                    )
                    # Stacking features for initial model
                    features = construct_tad_features(
                        curr_token_prob,
                        token_idx,
                        tad_probs,
                        train_attention_features,
                        attn_ptr,
                        train_greedy_log_likelihoods,
                        seq_idx,
                        self.n_previous_tokens,
                        self.n_attn,
                    )
                    # Predict using stacking model for this fold
                    fold_id = self.fold_idx[seq_idx]
                    pred_prob = self.stacking_models[fold_id].predict([features])[0]
                    pred_prob = np.clip(pred_prob, 0, 1)
                    tad_probs.append(pred_prob)
                    y_seq.append(seq_score)
                    # Stacking features for the final model using the initial model's predictions
                    features_step_2 = construct_tad_features(
                        curr_token_prob,
                        token_idx,
                        tad_probs,
                        train_attention_features,
                        attn_ptr,
                        train_greedy_log_likelihoods,
                        seq_idx,
                        self.n_previous_tokens,
                        self.n_attn,
                    )
                    X_seq.append(features_step_2)
                    attn_ptr += 1
                if X_seq:
                    X_stack.append(X_seq)
                    y_stack.append(y_seq)
            X_stack = np.concatenate(X_stack)
            y_stack = np.concatenate(y_stack)

            # Final model fit
            best_params = cross_val_hp(
                X_stack,
                y_stack,
                self.model_init,
                self.params,
                train_attention_features,
                train_greedy_log_likelihoods,
                train_metrics,
                n_previous_tokens=self.n_previous_tokens,
                step=2,
            )
            self.regression_model = self.model_init(best_params)
            self.regression_model.fit(X_stack, y_stack)
            self.aggregation = best_params[1]
            self.is_fitted = True

        # Inference
        tad_scores = []
        attn_ptr = 0
        for seq_idx in range(len(test_greedy_log_likelihoods)):
            tad_probs = [np.exp(test_greedy_log_likelihoods[seq_idx][0])]
            for token_idx in range(1, len(test_greedy_log_likelihoods[seq_idx])):
                curr_token_prob = np.exp(
                    test_greedy_log_likelihoods[seq_idx][token_idx]
                )
                features = construct_tad_features(
                    curr_token_prob,
                    token_idx,
                    tad_probs,
                    test_attention_features,
                    attn_ptr,
                    test_greedy_log_likelihoods,
                    seq_idx,
                    self.n_previous_tokens,
                    self.n_attn,
                )
                pred_prob = self.regression_model.predict([features])[0]
                pred_prob = np.clip(pred_prob, 0, 1)
                tad_probs.append(pred_prob)
                attn_ptr += 1
            tad_probs = np.array(tad_probs)
            if self.aggregation == "mean":
                tad_scores.append(-tad_probs.mean())
            elif self.aggregation == "sum(log(p_i))":
                tad_scores.append(-np.log(tad_probs + 1e-5).sum())
            else:
                raise ValueError(f"Aggregation {self.aggregation} not supported")
        return np.array(tad_scores)
