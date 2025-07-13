from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model

from typing import Dict, List, Tuple

import numpy as np

import torch


class InferCausalLMCalculator(StatCalculator):
    """
    Performs inference of the model and ensures that output contains
    1. logprobas
    2. tokens
    3. embeddings

    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    def __init__(
        self,
        n_alternatives: int = 10,
        tokenize: bool = False,
        return_embeddings: bool = True,
    ):
        super().__init__()

        self.n_alternatives = n_alternatives
        self._tokenize = tokenize
        self._return_embeddings = return_embeddings

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "greedy_log_probs",
            "greedy_logits",
            "greedy_tokens",
            "greedy_log_likelihoods",
            "greedy_tokens_alternatives",
            "embeddings_decoder",
        ], []

    def _get_embeddings_from_output(
        self,
        output,
        all_layers: bool = False,
    ):
        batch_embeddings_decoder = None

        if not all_layers:
            hidden_layer = -1
            input_tokens_hs = output.hidden_states[0][hidden_layer].cpu().detach()
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [h[hidden_layer].cpu().detach() for h in output.hidden_states[1:]],
                    dim=1,
                )
        else:
            input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [h.mean(axis=0).cpu().detach() for h in output.hidden_states[1:]],
                    dim=1,
                )

        if len(output.hidden_states) > 1:
            batch_embeddings_decoder = (
                torch.cat([input_tokens_hs, generated_tokens_hs], dim=1)
                .mean(axis=1)
                .cpu()
                .detach()
            )
        else:
            batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()

        return batch_embeddings_decoder

    def _post_process_logits(self, out, model_inputs, eos_token_id):
        cut_logits = []
        cut_sequences = []
        cut_log_probs = []
        cut_alternatives = []
        lls = []

        all_logits = torch.stack(out.scores, dim=1)
        for i in range(len(model_inputs)):
            seq = out.sequences[i, model_inputs.shape[1] :].cpu()

            length = len(seq)
            for j in range(len(seq)):
                if seq[j] == eos_token_id:
                    length = j + 1
                    break

            tokens = seq[:length].tolist()
            cut_sequences.append(tokens)

            logits = all_logits[i, :length, :].cpu()
            cut_logits.append(logits.numpy())

            log_probs = logits.log_softmax(-1)
            cut_log_probs.append(log_probs.numpy())
            lls.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[j, :].numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))

                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        result_dict = {
            "greedy_log_probs": cut_log_probs,
            "greedy_logits": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_log_likelihoods": lls,
            "greedy_tokens_alternatives": cut_alternatives,
        }

        return result_dict

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int,  # TODO: move to args_generate
        **kwargs,
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
                - 'input_texts' (List[str]): input texts batch,
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """

        model_inputs = (
            model.tokenize(texts, padding=True, return_tensors="pt")
            if self._tokenize
            else dependencies["model_inputs"]
        )

        input_ids = (
            model_inputs[0]
            if isinstance(model_inputs, tuple)
            else model_inputs["input_ids"]
        )

        args_generate = {
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_hidden_states": True,
            "max_new_tokens": max_new_tokens,
        }
        args_generate.update(kwargs)
        out = model.generate(**model_inputs, **args_generate)

        result_dict = self._post_process_logits(
            out, input_ids, model.model.generation_config.eos_token_id
        )

        if self._return_embeddings:
            result_dict.update(
                {"embeddings_decoder": self._get_embeddings_from_output(out)}
            )

        result_dict.update({"out": out})

        return result_dict
