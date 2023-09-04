import numpy as np

from typing import Dict

from .estimator import Estimator


class SeqEntropySeq(Estimator):
    def __init__(self, normalization_method: str, gen_max_len: int):
        super().__init__(['sample_log_probs'], 'sequence')

        self.normalization_method = normalization_method
        self.gen_max_len = gen_max_len

    def __str__(self):
        return 'SeqEntropy'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_logprobs = stats['sample_log_probs']
        batch_tokens = stats['sample_tokens']
        
        batch_entropy_estimates = []
        for hypos_logprobs, hypos in zip(batch_logprobs, batch_tokens):
            norm_logprobs = []
            importance_weights = []
            for hyp_lp, hyp_tokens in zip(hypos_logprobs, hypos):
                norm_logprob = hyp_lp / len(hyp_tokens)
                weight = np.exp(norm_logprob - hyp_lp)
                norm_logprobs.append(norm_logprob)
                importance_weights.append(weight)
            
            _, unique_seq_ids = np.unique(hypos, axis=0, return_index=True)

            unique_seq_lps = np.array(hypos_logprobs)[unique_seq_ids]
            unique_seq_norm_lps = np.array(norm_logprobs)[unique_seq_ids]

            base_norm_constant = np.sum(unique_seq_norm_lps)
            lp_rem = 1 - np.sum(unique_seq_lps)
            match normalization_method:
                case 'lower_bound':
                    add_term = lp_rem
                case 'upper_bound':
                    add_term = np.exp(np.log(lp_rem) / gen_max_len)
                case 'mean':
                    add_term = np.mean(lp_rem, np.exp(np.log(lp_rem) / self.gen_max_len))
                case _:
                    raise Exception(f'Unknown normalization method: {normalization_method}')

            norm_constant_estimate = base_norm_constant + add_term
            
            summand = [weight * (norm_lp - norm_constant_estimate) for weight, norm_lp in zip(norm_logprobs, importance_weights)]

            entropy_estimate = np.mean(summand) / norm_constant_estimate
            batch_entropy_estimates.append(entropy_estimate)

        return batch_entropy_estimates
