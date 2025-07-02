import torch
import numpy as np

from typing import Dict, List, Tuple, Union
from collections import deque

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "input_texts",
            "input_tokens",
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "embeddings",
            "attention_all",
            "tokenizer",
        ], []

    def __init__(
        self,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
        n_alternatives: int = 10,
        answer_marker: str = "### Answer:",
        slicing_target: str = None,
    ):
        """
        Initializes the calculator.

        Parameters:
            output_attentions (bool): Whether to calculate and return attention scores.
            output_hidden_states (bool): Whether to calculate and return embeddings.
            n_alternatives (int): Number of alternative tokens to store at each position.
            answer_marker (str): The string that separates different parts of the generation.
            slicing_target (str): Determines which part of the generation to analyze.
                - "answer": Analyzes the text after the answer_marker.
                - "reasoning": Analyzes the text before the answer_marke`.
                - None or any other string: Analyzes the full generation without slicing.
        """
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.n_alternatives = n_alternatives
        
        if slicing_target not in ["answer", "reasoning", None]:
            self.slicing_target = None
        else:
            self.slicing_target = slicing_target
        self.answer_marker = answer_marker if self.slicing_target else None


    def _find_token_subsequence(self, main_list: List[int], len_sub: int, substring: str, tokenizer) -> int:
        """
        Finds the starting index of a token sublist in the main list by comparing the decoded strings.
        """
        if len_sub == 0 or len_sub > len(main_list):
            return -1
        initial_window_tokens = main_list[len(main_list) - len_sub:]
        window_decoded = deque(
            (tokenizer.decode(t) for t in initial_window_tokens), maxlen=len_sub
        )

        for i in range(len(main_list) - len_sub, -1, -1):
            current_window_text = "".join(list(window_decoded))
            if substring in current_window_text:
                return i
            if i > 0:
                new_token = main_list[i - 1]
                new_decoded_token = tokenizer.decode(new_token)
                window_decoded.appendleft(new_decoded_token)

        return -1


    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
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
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
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
            if model.model_type == "vLLMCausalLM":
                logits = logits.transpose(1, 0)
            sequences = out.sequences
            if self.output_attentions:
                attentions = out.attentions
            if self.output_hidden_states:
                embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                    out, batch, model.model_type
                )
                if embeddings_decoder.dtype == torch.bfloat16:
                    embeddings_decoder = embeddings_decoder.to(torch.float16)

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        all_slice_start_indices = []

        marker_tokens = []
        if self.answer_marker:
            marker_tokens = model.tokenizer(
                self.answer_marker, add_special_tokens=False
            ).input_ids

        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                full_gen_seq = sequences[i, idx:].cpu()
            elif model.model_type == "vLLMCausalLM":
                full_gen_seq = sequences[i].cpu()
            else:
                full_gen_seq = sequences[i, 1:].cpu()

            slice_start_idx = 0
            slice_end_idx = len(full_gen_seq)
            marker_pos = -1

            if self.slicing_target and len(marker_tokens) > 0:
                marker_pos = self._find_token_subsequence(full_gen_seq.tolist(), len(marker_tokens), self.answer_marker, model.tokenizer)

            if self.slicing_target == "answer":
                if marker_pos != -1:
                    slice_start_idx = marker_pos + len(marker_tokens)
                else:
                    # If marker not found for answer mode, produces empty result
                    slice_start_idx, slice_end_idx = 0, 0
            elif self.slicing_target == "reasoning":
                if marker_pos != -1:
                    slice_end_idx = marker_pos
                # If marker not found, process whole sequence


            all_slice_start_indices.append(slice_start_idx)
            seq = full_gen_seq[slice_start_idx:slice_end_idx]

            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break

            final_seq_tokens = seq[:length].tolist()
            final_seq_text_tokens = seq[:text_length]

            cut_sequences.append(final_seq_tokens)
            cut_texts.append(model.tokenizer.decode(final_seq_text_tokens))

            cut_logits.append(logits[i, slice_start_idx : slice_start_idx + length, :].cpu().numpy())

            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                # Absolute offset
                lt = logits[i, j + slice_start_idx, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: str(x)[0] == str(final_seq_tokens[j]),
                    reverse=True,
                )

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            if len(tokens) == 0:
                ll.append([])
                continue
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        attention_all = []

        if self.output_attentions and (model.model_type != "vLLMCausalLM"):
            config = model.model.config
            if hasattr(config, 'text_config'):
                config = config.text_config
            for i in range(len(texts)):
                slice_start_idx = all_slice_start_indices[i]
                c = len(cut_sequences[i])
                attn_mask = np.zeros(shape=(
                    config.num_attention_heads * config.num_hidden_layers, c, c
                ))
                
                if c == 0: # Empty sequence
                    attention_all.append(attn_mask.max(0))
                    continue

                for j in range(1, c):
                    original_token_index = j + slice_start_idx
                    if original_token_index < len(attentions):
                        stacked_attention = torch.vstack([
                            attentions[original_token_index][layer][0][head][0][-j:]
                            for layer in range(len(attentions[original_token_index]))
                            for head in range(len(attentions[original_token_index][layer][0]))
                        ])
                        if stacked_attention.dtype == torch.bfloat16:
                            stacked_attention = stacked_attention.to(torch.float16)
                        attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all.append(attn_mask.max(0))

        if not self.output_hidden_states:
            embeddings_dict = {}
        elif model.model_type == "CausalLM":
            embeddings_dict = {
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        elif model.model_type == "Seq2SeqLM":
            embeddings_dict = {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
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
        }
        result_dict.update(embeddings_dict)
        if self.output_attentions:
            result_dict.update({"attention_all": attention_all})
            result_dict.update({"tokenizer": model.tokenizer})
        return result_dict
