import numpy as np
import torch

from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical

TOP_K = [5, 10, 15]


def get_collect_fn(model_output):
    if type(model_output).__name__ == "SampleEncoderDecoderOutput":
        return collect_sample_token_level_uncertainties
    return collect_token_level_uncertainties


def collect_sample_token_level_uncertainties(
    model_output,
    batch_size,
    num_return_sequences,
    vocab_size,
    pad_token_id,
    length_penalty=1.0,
    ensemble_uncertainties={},
):
    base_shape = [batch_size, num_return_sequences]
    seq_length = model_output["sequences"].shape[-1]

    seq_shape = base_shape + [seq_length]
    sequences = model_output["sequences"].reshape(seq_shape)[:, :, 1:]
    # 0 - iters
    # 1 - num_obs * num_ret_seq
    # 2 - vocab_size
    scores = torch.stack(model_output.generation_scores).permute(1, 0, 2)
    scores_shape = base_shape + [seq_length - 1, vocab_size]
    scores = scores.reshape(scores_shape)
    device = scores.device

    token_scores = torch.zeros(base_shape + [seq_length - 1]).to(device)

    token_measures = (
        list(ensemble_uncertainties.keys())
        + [f"entropy_top{k}" for k in TOP_K]
        + ["entropy"]
    )

    unc_shape = base_shape + [seq_length - 1]
    token_level_uncertainties = {key: torch.zeros(unc_shape) for key in token_measures}

    output_uncertainties_reshaped = {
        key: torch.stack(ensemble_uncertainties[key], dim=-1).reshape(unc_shape)
        for key in ensemble_uncertainties.keys()
    }

    aggregate_models = (
        "models_scores" in model_output and len(model_output["models_scores"]) > 0
    )

    if aggregate_models:
        num_models = len(model_output["models_scores"][0])
        models_sequence_scores = torch.zeros(
            batch_size, num_models, num_return_sequences, seq_length
        )

    seq_lengths = (model_output["sequences"] != pad_token_id).sum(dim=-1)
    seq_lengths = seq_lengths.reshape(base_shape).to(device)

    seq_penalty = seq_lengths**length_penalty
    seq_penalty_unb = (seq_lengths - 1) ** length_penalty

    for obs_id in range(batch_size):
        for _iter in reversed(range(sequences.shape[-1])):
            for seq_i in range(num_return_sequences):
                index = (obs_id, seq_i, _iter)
                token = sequences[index]
                if token == pad_token_id:
                    continue
                else:
                    posterior_logs = log_softmax(scores[index], dim=-1)
                    token_scores[index] = posterior_logs[token]
                    posterior = posterior_logs.exp()

                    if aggregate_models:
                        for i, model_logits in enumerate(
                            model_output["models_scores"][_iter]
                        ):
                            model_logits = model_logits.reshape(
                                batch_size, num_return_sequences, vocab_size
                            )
                            models_sequence_scores[obs_id, i, seq_i, _iter] = (
                                model_logits[obs_id, seq_i, token]
                            )

                    entropies = {}
                    entropies["entropy"] = Categorical(posterior).entropy()
                    entropies["entropy_top5"] = Categorical(
                        posterior.topk(5, dim=-1).values
                    ).entropy()
                    entropies["entropy_top10"] = Categorical(
                        posterior.topk(10, dim=-1).values
                    ).entropy()
                    entropies["entropy_top15"] = Categorical(
                        posterior.topk(15, dim=-1).values
                    ).entropy()
                    for key in token_measures:
                        if key in [
                            "entropy",
                            "entropy_top5",
                            "entropy_top10",
                            "entropy_top15",
                        ]:
                            ue = entropies[key]
                        else:
                            ue = output_uncertainties_reshaped[key][index]
                        token_level_uncertainties[key][index] = torch.tensor(ue)

    sequences_scores = token_scores.sum(dim=-1) / seq_penalty
    entropy_s = Categorical(sequences_scores.exp())

    if aggregate_models:
        models_sequence_scores = (
            models_sequence_scores.sum(dim=-1).to(device) / seq_penalty
        )
        token_level_uncertainties["log_probas"] = models_sequence_scores
        token_level_uncertainties["probas"] = models_sequence_scores.exp()

    for key in token_measures:
        token_level_uncertainties[key] = (
            token_level_uncertainties[key].sum(dim=-1).to(device)
        )
        token_level_uncertainties[key] = (
            token_level_uncertainties[key] / seq_penalty_unb
        )

    beam_weights = sequences_scores.exp() / sequences_scores.exp().sum(
        dim=-1, keepdim=True
    )
    token_level_uncertainties["beam_weights"] = beam_weights

    beam_scores_unb = sequences_scores * seq_penalty / seq_penalty_unb
    entropy_s_u = Categorical(sequences_scores.exp())

    token_level_uncertainties["scores_unbiased"] = beam_scores_unb
    beam_weights_unb = beam_scores_unb.exp() / beam_scores_unb.exp().sum(
        dim=-1, keepdim=True
    )
    token_level_uncertainties["weights"] = beam_weights_unb

    token_level_uncertainties["sequences_scores"] = sequences_scores.cpu().reshape(
        batch_size * num_return_sequences
    )
    token_level_uncertainties["entropy_s"] = entropy_s
    token_level_uncertainties["entropy_s_u"] = entropy_s_u

    for key in token_level_uncertainties.keys():
        token_level_uncertainties[key] = token_level_uncertainties[key].cpu().numpy()

    return token_level_uncertainties


def collect_token_level_uncertainties(
    model_output,
    batch_size,
    beam_size,
    vocab_size,
    pad_token_id,
    length_penalty=1.0,
    ensemble_uncertainties={},
):
    beam_ids = model_output["beam_indices"]
    seq_len = beam_ids.shape[-1]
    shape = (batch_size, beam_size, seq_len)
    beam_ids = beam_ids.reshape(shape)
    beam_ids = beam_ids[:, :, :-1]
    beam_ids_finished_mask = beam_ids == -1
    beam_ids = beam_ids % beam_size
    beam_ids[beam_ids_finished_mask] = -1

    token_measures = (
        list(ensemble_uncertainties.keys())
        + [f"entropy_top{k}" for k in TOP_K]
        + ["entropy"]
    )

    token_level_uncertainties = {key: torch.zeros(shape) for key in token_measures}

    aggregate_models = (
        "models_scores" in model_output and len(model_output["models_scores"]) > 0
    )

    if aggregate_models:
        num_models = len(model_output["models_scores"][0])
        models_sequence_scores = torch.zeros(batch_size, num_models, beam_size, seq_len)

    # For some reason, beam search can truncate generation iterations, so
    # seq len from beam_ids can be less than iterations steps number
    unc_length = len(model_output.generation_scores)
    unc_shape = (batch_size, beam_size, unc_length)
    output_uncertainties_reshaped = {
        key: torch.stack(ensemble_uncertainties[key], dim=-1).reshape(unc_shape)
        for key in ensemble_uncertainties.keys()
    }

    device = beam_ids.device
    seq_lengths = (model_output["sequences"] != pad_token_id).sum(dim=-1)
    seq_lengths = seq_lengths.reshape(batch_size, beam_size).to(device)

    seq_penalty = seq_lengths**length_penalty
    seq_penalty_unb = (seq_lengths - 1) ** length_penalty

    sequences = model_output["sequences"].reshape(shape)[:, :, 1:]

    for obs_id in range(batch_size):
        for _iter in reversed(range(beam_ids.shape[-1])):
            iter_beam_ids = beam_ids[obs_id, :, _iter]
            for seq_i, beam_id in enumerate(iter_beam_ids):
                if beam_id == -1:
                    continue
                else:
                    posterior = (
                        model_output.generation_scores[_iter]
                        .reshape(batch_size, beam_size, vocab_size)[obs_id, beam_id]
                        .exp()
                    )
                    if aggregate_models:
                        token = sequences[obs_id, seq_i, _iter]
                        for i, model_logits in enumerate(
                            model_output["models_scores"][_iter]
                        ):
                            model_logits = model_logits.reshape(
                                batch_size, beam_size, vocab_size
                            )
                            models_sequence_scores[obs_id, i, seq_i, _iter] = (
                                model_logits[obs_id, beam_id, token]
                            )

                    entropies = {}
                    entropies["entropy"] = Categorical(posterior).entropy()
                    entropies["entropy_top5"] = Categorical(
                        posterior.topk(5, dim=-1).values
                    ).entropy()
                    entropies["entropy_top10"] = Categorical(
                        posterior.topk(10, dim=-1).values
                    ).entropy()
                    entropies["entropy_top15"] = Categorical(
                        posterior.topk(15, dim=-1).values
                    ).entropy()
                    for key in token_measures:
                        if key in [
                            "entropy",
                            "entropy_top5",
                            "entropy_top10",
                            "entropy_top15",
                        ]:
                            ue = entropies[key]
                        else:
                            ue = output_uncertainties_reshaped[key][
                                obs_id, beam_id, _iter
                            ]
                        token_level_uncertainties[key][obs_id, seq_i, _iter] = (
                            torch.tensor(ue)
                        )

    for key in token_measures:
        token_level_uncertainties[key] = (
            token_level_uncertainties[key].sum(dim=-1).to(device)
        )
        token_level_uncertainties[key] = (
            token_level_uncertainties[key] / seq_penalty_unb
        )

    if aggregate_models:
        modelwise_penalties = seq_penalty.unsqueeze(1).repeat(
            1, models_sequence_scores.shape[1], 1
        )
        models_sequence_scores = (
            models_sequence_scores.sum(dim=-1).to(device) / modelwise_penalties
        )

        token_level_uncertainties["log_probas"] = models_sequence_scores
        token_level_uncertainties["probas"] = models_sequence_scores.exp()

    beam_scores = model_output["sequences_scores"].reshape(batch_size, beam_size)
    entropy_s = Categorical(beam_scores.exp()).entropy()
    beam_weights = beam_scores.exp() / beam_scores.exp().sum(dim=-1, keepdim=True)
    token_level_uncertainties["beam_weights"] = beam_weights

    beam_scores_unb = beam_scores * (seq_penalty / seq_penalty_unb)
    entropy_s_u = Categorical(beam_scores_unb.exp()).entropy()
    token_level_uncertainties["scores_unbiased"] = beam_scores_unb
    beam_weights_unb = beam_scores_unb.exp() / beam_scores_unb.exp().sum(
        dim=-1, keepdim=True
    )
    token_level_uncertainties["weights"] = beam_weights_unb

    token_level_uncertainties["entropy_s"] = entropy_s
    token_level_uncertainties["entropy_s_u"] = entropy_s_u

    for key in token_level_uncertainties.keys():
        token_level_uncertainties[key] = token_level_uncertainties[key].cpu().numpy()

    return token_level_uncertainties


def update_token_level_scores(scores, batch_scores):
    for key in scores:
        if scores[key] is None:
            scores[key] = batch_scores[key]
        else:
            scores[key] = np.r_[scores[key], batch_scores[key]]
    return scores
