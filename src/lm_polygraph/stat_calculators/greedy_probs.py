import torch
import numpy as np

from typing import Dict, List

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class FeatureExtractorLookbackLens:
    def __init__(self, orig_base_model, **kwargs):
        self._n_layers = orig_base_model.config.num_hidden_layers
        self._n_heads = orig_base_model.config.num_attention_heads
        self._input_size = self._n_layers * self._n_heads

    def feature_dim(self):
        return self._input_size

    def __call__(self, llm_inputs, llm_outputs):
        attentions_all = llm_outputs.attentions

        is_training = not hasattr(llm_outputs, "sequences")
        layer = 0  # only for tensor shapes
        batch_sz = llm_inputs['attention_mask'].shape[0]

        if is_training:
            context_bounds = torch.as_tensor(llm_outputs.context_lengths, device=attentions_all[0].device)
            attentions_all = [
                tuple(
                    attentions_all[l][:, :, i:i + 1, :i + 1]
                    for l in range(len(attentions_all))
                )
                for i in range(attentions_all[layer].shape[-1] - 1)
            ]
        else:
            context_bounds = torch.tensor(
                [attentions_all[0][layer].shape[-1] for _ in range(batch_sz)], device=attentions_all[0][0].device)
            attn_inp = [
                tuple(
                    attentions_all[0][l][:, :, i:i + 1, :i + 1]
                    for l in range(len(attentions_all[0]))
                )
                for i in range(attentions_all[0][layer].shape[-2])
            ]
            inp_len = attentions_all[0][layer].shape[-2]
            outp_len = len(attentions_all[1:])
            attn_outp = [
                tuple(
                    a[l][:, :, :, :i + 1]
                    for l in range(len(attentions_all[0]))
                )
                for i, a in zip(range(inp_len, inp_len + outp_len), attentions_all[1:])
            ]
            attentions_all = attn_inp + attn_outp

        context_lengths = [
            llm_inputs['attention_mask'][i][:context_bounds[i]].sum().item()
            for i in range(batch_sz)
        ]

        all_features = []
        for seq_idx, attentions in enumerate(attentions_all):
            features = []
            assert attentions[0].shape[2] == 1

            attn_ctx, attn_new = [], []
            for l in range(self._n_layers):
                a = attentions[l]  # shape: (batch_sz, H, 1, seq_len)
                ctx_bounds = context_bounds  # shape: (batch_sz)
                seq_range = torch.arange(a.shape[-1], device=a.device).unsqueeze(0)  # shape: (1, seq_len)
                ctx_mask = seq_range < ctx_bounds.unsqueeze(1)  # shape: (batch_sz, seq_len)
                new_mask = ~ctx_mask  # Complement of the context mask
                attn_ctx_layer = (a[:, :, 0, :] * ctx_mask.unsqueeze(1)).sum(-1)
                attn_new_layer = (a[:, :, 0, :] * new_mask.unsqueeze(1)).sum(-1)
                attn_ctx.append(attn_ctx_layer)
                attn_new.append(attn_new_layer)
            attn_ctx = torch.stack(attn_ctx)  # shape: (L, batch_sz, H)
            attn_new = torch.stack(attn_new)  # shape: (L, batch_sz, H)

            for batch_i in range(batch_sz):
                # calculate input length
                ctx_len = context_lengths[batch_i]
                ctx_bound = context_bounds[batch_i]
                if seq_idx > ctx_bound:  # in the new tokens
                    mean_attn_ctx = attn_ctx[:, batch_i, :] / ctx_len
                    mean_attn_new = attn_new[:, batch_i, :] / (seq_idx - ctx_bound)
                    lb_ratio = mean_attn_new / (mean_attn_ctx + mean_attn_new)
                else:  # in the padding / context
                    lb_ratio = torch.ones_like(attn_ctx[:, batch_i, :])
                features.append(lb_ratio.reshape(-1))
            features = torch.stack(features)  # batch_size x feature_vector
            all_features.append(features)

        # Output: batch_size x sequence_length x feature_vector
        result = torch.stack(all_features, dim=1)

        return result

    def input_size(self):
        return self._input_size

    def output_attention(self):
        return True


def load_extractor(config, base_model, *args, **kwargs):
    return FeatureExtractorLookbackLens(base_model)

class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    Calculates generation texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self):
        super().__init__(["greedy_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        with torch.no_grad():
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
            )

        return {"greedy_texts": sequences}


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    def __init__(self, n_alternatives: int = 10):
        super().__init__(
            [
                "input_tokens",
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_tokens_alternatives",
                "greedy_texts",
                "greedy_log_likelihoods",
                "train_greedy_log_likelihoods",
                "embeddings",
                "lookback_lens_features",
                "all_layers_embeddings",
            ],
            [],
        )
        self.n_alternatives = n_alternatives

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        lbl_extractor = FeatureExtractorLookbackLens(model.model)

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=True,
                output_hidden_states=True,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            logits = torch.stack(out.scores, dim=1)

            sequences = out.sequences
            embeddings_encoder, embeddings_decoder, all_layers_embeddings = get_embeddings_from_output(
                out,
                batch,
                model.model_type,
                model.tokenizer.config.pad_token_id,
                save_all_embeddings=True
            )
            lbl = lbl_extractor(batch, out)

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                seq = sequences[i, idx:].cpu()
            else:
                seq = sequences[i, 1:].cpu()
            length = len(seq)
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:length], skip_special_tokens=True))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        if model.model_type == "CausalLM":
            embeddings_dict = {
                "embeddings_decoder": embeddings_decoder,
                "all_layers_embeddings": all_layers_embeddings,
            }
        elif model.model_type == "Seq2SeqLM":
            embeddings_dict = {
                "embeddings_encoder": embeddings_encoder,
                "embeddings_decoder": embeddings_decoder,
            }
        else:
            raise NotImplementedError

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
            "lookback_lens_features": lbl,
        }
        result_dict.update(embeddings_dict)

        return result_dict
