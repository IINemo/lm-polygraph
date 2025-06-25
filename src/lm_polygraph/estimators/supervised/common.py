import logging
import itertools
from typing import Callable, Any

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("lm_polygraph")


def cross_val_hp(
    X: np.ndarray,
    y: np.ndarray,
    model_init: Callable[[Any], nn.Module],
    params: dict,
    mask: np.ndarray = None,
    regression: bool = False,
    estimator_name: str = "SAPLMA",
):
    if regression:
        best_score = np.inf
        metric = mean_squared_error
    else:
        best_score = -np.inf
        metric = roc_auc_score

    best_params = None
    param_grid = list(itertools.product(*params.values()))
    for param in tqdm(param_grid, desc=f"Hyperparameter search for {estimator_name}"):
        model = model_init(param)
        scores_cv = []
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        for i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(X)))):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            if mask is not None:
                mask_train, mask_val = mask[train_idx], mask[val_idx]
            else:
                mask_train, mask_val = None, None

            if mask_train is not None:
                model.fit(X_train, y_train, mask_train, regression=regression)
            else:
                model.fit(X_train, y_train, regression=regression)
            try:
                if mask_val is not None:
                    y_pred = model.predict(X_val, mask_val, regression=regression)
                else:
                    y_pred = model.predict(X_val, regression=regression)
                scores_cv.append(metric(y_val, y_pred))
            except Exception as e:
                log.info(
                    f"Skip fold {i} in cross-validation in the {estimator_name} estimator with error: {e}"
                )

        if scores_cv:
            scores_mean = np.mean(scores_cv)
        elif regression:
            scores_mean = np.inf
        else:
            scores_mean = -np.inf

        if regression:
            if scores_mean < best_score:
                best_score = scores_mean
                best_params = param
        else:
            if best_score < scores_mean:
                best_score = scores_mean
                best_params = param
    log.info(
        f"{estimator_name} best hyperparameters: {best_params}, best score: {best_score}"
    )
    if best_params is None:
        best_params = param_grid[0]
    return best_params


class TrainerMLP:
    def __init__(
        self,
        n_epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.001,
        n_features: int = 4096,
        device: str = "cuda",
        loss_fn: nn.Module = None,
        model: nn.Module = None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = model(n_features)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def fit(self, X, y, mask=None, regression: bool = False):
        self.model.train()
        X_torch = (
            torch.tensor(X, dtype=torch.float32)
            if not isinstance(X, torch.Tensor)
            else X.clone().detach().float()
        )
        y_torch = (
            torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            if not isinstance(y, torch.Tensor)
            else y.clone().detach().float()
        )
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_torch = y_torch.long().squeeze(1)
        if mask is not None:
            mask_torch = (
                torch.tensor(mask, dtype=torch.bool)
                if not isinstance(mask, torch.Tensor)
                else mask.clone().detach().bool()
            ).unsqueeze(2)
        batch_indices = torch.arange(0, len(X), self.batch_size)
        self.model.to(self.device)
        for epoch in range(self.n_epochs):
            for start in batch_indices:
                X_batch = X_torch[start : start + self.batch_size].to(self.device)
                y_batch = y_torch[start : start + self.batch_size].to(self.device)
                if mask is not None:
                    mask_batch = mask_torch[start : start + self.batch_size].to(
                        self.device
                    )
                    y_pred = self.model(X_batch, mask_batch, regression=regression)
                else:
                    y_pred = self.model(X_batch, regression=regression)
                try:
                    loss = self.loss_fn(y_pred, y_batch)
                except Exception as e:
                    print(y_pred, y_batch)
                    raise e
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X, mask=None, regression: bool = False):
        X_torch = (
            torch.tensor(X, dtype=torch.float32)
            if not isinstance(X, torch.Tensor)
            else X.clone().detach().float()
        )
        if mask is not None:
            mask_torch = (
                torch.tensor(mask, dtype=torch.bool)
                if not isinstance(mask, torch.Tensor)
                else mask.clone().detach().bool()
            ).unsqueeze(2)
        batch_indices = torch.arange(0, len(X), self.batch_size)
        self.model.eval()
        if next(self.model.parameters()).device.type != self.device:
            self.model.to(self.device)
        predictions = []
        with torch.no_grad():
            for start in batch_indices:
                X_batch = X_torch[start : start + self.batch_size].to(self.device)
                if mask is not None:
                    mask_batch = mask_torch[start : start + self.batch_size].to(
                        self.device
                    )
                    y_pred = self.model(
                        X_batch, mask_batch, regression=regression, eval=True
                    )
                else:
                    y_pred = self.model(X_batch, regression=regression, eval=True)
                predictions.append(y_pred.cpu().flatten())
        return np.concatenate(predictions)
