import numpy as np
import torch

from typing import Dict

from .estimator import Estimator


def get_seq_level_ue(sequence_level_data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    softmax_t = 1
    model_log_probas = sequence_level_data['log_probas'] # num_obs x num_models x num_beams
    ens_log_probas = torch.tensor(model_log_probas).logsumexp(1) - torch.tensor(model_log_probas.shape[1]).log()  # num_obs x num_beams
    ens_log_probas = ens_log_probas.numpy()
    ens_probas = np.exp(ens_log_probas)

    ens_probas_exp = ens_probas**softmax_t
    weights = ens_probas_exp / ens_probas_exp.sum(-1, keepdims=True)

    tu = (ens_probas * weights).sum(-1)  # num_obs
    
    rmi = (
        ((ens_log_probas - model_log_probas) * weights[None, :, :]).sum(-1).mean(1)
    )  # num_obs
    rmi_abs = (
        (np.abs(ens_log_probas - model_log_probas) * weights[None, :, :])
        .sum(-1)
        .mean(1)
    )  # num_obs

    uncertainty_estimates = {
        f"tu": -tu,
        f"rmi": rmi,
        f"rmi-abs": rmi_abs,
    }

    return uncertainty_estimates


class EPStu(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'EPStu'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['ep_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['tu']


class EPSrmi(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'EPSrmi'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['ep_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['rmi']


class EPSrmiabs(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'EPSrmiabs'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['ep_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['rmi-abs']


class PEStu(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'PEStu'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['pe_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['tu']


class PESrmi(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'PESrmi'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['pe_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['rmi']


class PESrmiabs(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'PESrmiabs'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sequence_level_data = stats['ensemble_token_scores']['pe_token_level_scores']

        return get_seq_level_ue(sequence_level_data)['rmi-abs']
