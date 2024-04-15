from .stat_calculator import StatCalculatorBasic

from typing import Dict

import numpy as np

import torch


class BasicGreedyProbsCalculatorCausalLM(StatCalculatorBasic):
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

    def __init__(self):
        super().__init__(
            [
                "input_tokens",
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_texts",
                "greedy_log_likelihoods",
                "train_greedy_log_likelihoods",
                "embeddings",
            ],
            [],
        )

    def get_embeddings_from_output(
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

    def __call__(
        self,
        model_inputs,
        model,
        args_generate,
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

        args_generate.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }
        )
        model_inputs = (
            model_inputs
            if isinstance(model_inputs, torch.Tensor)
            else model_inputs["input_ids"]
        )

        out = model.generate(model_inputs, **args_generate)

        all_logits = torch.stack(out.scores, dim=1)

        generation_config = args_generate["generation_config"]

        cut_logits = []
        cut_sequences = []
        cut_log_probs = []
        lls = []
        for i in range(len(model_inputs)):
            seq = out.sequences[i, model_inputs.shape[1] :].cpu()

            length = len(seq)
            for j in range(len(seq)):
                if seq[j] == generation_config.eos_token_id:
                    length = j + 1
                    break

            tokens = seq[:length].tolist()
            cut_sequences.append(tokens)

            logits = all_logits[i, :length, :].cpu()
            cut_logits.append(logits.numpy())

            log_probs = logits.log_softmax(-1)
            cut_log_probs.append(log_probs.numpy())
            lls.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        embeddings_decoder = self.get_embeddings_from_output(out)

        result_dict = {
            "greedy_log_probs": cut_log_probs,
            "greedy_logits": cut_logits,
            "greedy_tokens": cut_sequences,
            "embeddings_decoder": embeddings_decoder,
            "greedy_log_likelihoods": lls,
        }

        return result_dict
