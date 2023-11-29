import numpy as np
import torch

from typing import Dict

from .estimator import Estimator


def all_ep_estimators():
    return [EPStu(), EPSrmi()]


def all_pe_estimators():
    return [PEStu(), PESrmi()]


def get_seq_level_ue(
    sequence_level_data: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    softmax_t = 1
    model_log_probas = sequence_level_data[
        "log_probas"
    ]  # num_obs x num_models x num_beams
    num_models = model_log_probas.shape[1]
    ens_log_probas = (
        torch.tensor(model_log_probas).logsumexp(1) - torch.tensor(num_models).log()
    )  # num_obs x num_beams
    ens_log_probas = ens_log_probas.numpy()
    ens_probas = np.exp(ens_log_probas)

    ens_probas_exp = ens_probas**softmax_t
    weights = ens_probas_exp / ens_probas_exp.sum(-1, keepdims=True)

    tu = (ens_probas * weights).sum(-1)  # num_obs

    ens_log_probas = np.repeat(ens_log_probas[:, None, :], num_models, axis=1)
    rmi_weights = np.repeat(weights[:, None, :], num_models, axis=1)
    rmi_base = (ens_log_probas - model_log_probas) * rmi_weights
    rmi = rmi_base.sum(-1).mean(1)  # num_obs
    rmi_abs = np.abs(rmi_base).sum(-1).mean(1)  # num_obs

    uncertainty_estimates = {
        "tu": -tu,
        "rmi": rmi,
        "rmi-abs": rmi_abs,
    }

    return uncertainty_estimates


class EPStu(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "EPStu"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["ep_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["tu"]


class EPSrmi(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "EPSrmi"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["ep_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["rmi"]


class EPSrmiabs(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "EPSrmiabs"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["ep_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["rmi-abs"]


class PEStu(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "PEStu"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["pe_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["tu"]


class PESrmi(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "PESrmi"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["pe_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["rmi"]


class PESrmiabs(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        return "PESrmiabs"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats["ensemble_token_scores"]["pe_token_level_scores"]

        return get_seq_level_ue(sequence_level_data)["rmi-abs"]
