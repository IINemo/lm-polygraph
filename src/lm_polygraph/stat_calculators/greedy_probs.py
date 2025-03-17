import torch
import numpy as np

from typing import Dict, List, Tuple

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    Calculates generation texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["greedy_texts"], []

    def __init__(self):
        super().__init__()

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
        sequences = model.generate_texts(
            input_texts=texts,
            max_new_tokens=max_new_tokens,
            n=1,
        )

        return {"greedy_texts": sequences}


class GreyboxGreedyProbsCalculator(StatCalculator):
    """
    Calculates generation texts and log probabilities for BlackboxModel that supports logprobs
    (such as OpenAI models via their API). This calculator enables "greybox" behavior, where
    we have partial access to model internals through the API.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "greedy_texts",
            "greedy_log_probs",
            "greedy_log_likelihoods",
            "greedy_tokens",
        ], []

    def __init__(
        self,
        top_logprobs: int = 5,
    ):
        super().__init__()
        self.top_logprobs = top_logprobs

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts and log probabilities for BlackboxModel supporting logprobs.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (BlackboxModel): Model used for generation (must have supports_logprobs=True).
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_log_probs' (List[List[np.array]]): logits for the top k tokens,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens,
                - 'greedy_tokens' (List[List[str]]): tokens of the generated text.
        """
        if not model.supports_logprobs:
            raise ValueError(
                "Model must support logprobs for GreyboxGreedyProbsCalculator"
            )

        # Request text generation with logprobs
        sequences = model.generate_texts(
            input_texts=texts,
            max_new_tokens=max_new_tokens,
            n=1,
            output_scores=True,
            top_logprobs=self.top_logprobs,
        )

        # Process the results to match the expected format for downstream estimators
        greedy_texts = sequences
        greedy_log_probs = []
        greedy_log_likelihoods = []
        greedy_tokens = []

        # Extract logprobs and tokens from the model's stored data
        if hasattr(model, "logprobs") and model.logprobs:
            for i, logprob_data in enumerate(model.logprobs):
                if hasattr(logprob_data, "content"):
                    # Extract tokens
                    tokens = [item.token for item in logprob_data.content]
                    greedy_tokens.append(tokens)

                    # Extract log probabilities for this generation
                    log_probs_list = []
                    log_likelihoods = []

                    for token_logprobs in logprob_data.content:
                        # Get the top logprobs for this token position from OpenAI API
                        token_logprob_dict = {}
                        for top_logprob in getattr(token_logprobs, "top_logprobs", []):
                            token_logprob_dict[top_logprob.token] = top_logprob.logprob

                        # Create a sparse representation of the logprobs distribution
                        # Note: Many uncertainty estimators like MaximumSequenceProbability, Perplexity,
                        # and MaximumTokenProbability don't use this full distribution - they only
                        # use greedy_log_likelihoods for the chosen tokens.
                        # However, we provide it for estimators that need the full distribution (e.g., entropy-based methods)
                        sparse_logprobs = np.ones(
                            model.model_path_vocab_size
                            if hasattr(model, "model_path_vocab_size")
                            else 50000
                        ) * -float("inf")

                        # Map token strings to positions in the sparse array
                        # This is an approximation since we don't have access to OpenAI's actual token IDs
                        # It only affects estimators that use the full probability distribution,
                        # not ones that just use the logprob of the chosen token
                        for token_str, logprob in token_logprob_dict.items():
                            token_idx = hash(token_str) % len(sparse_logprobs)
                            sparse_logprobs[token_idx] = logprob

                        log_probs_list.append(sparse_logprobs)

                        # Extract the log probability of the chosen token
                        # This is what's used by MaximumSequenceProbability, Perplexity and similar estimators
                        # and is directly provided by the OpenAI API without any mapping needed
                        chosen_logprob = token_logprobs.logprob
                        log_likelihoods.append(chosen_logprob)

                    greedy_log_probs.append(log_probs_list)
                    greedy_log_likelihoods.append(log_likelihoods)

        # Ensure all outputs have the same length
        while len(greedy_tokens) < len(greedy_texts):
            # If we're missing token data, add placeholder empty lists
            greedy_tokens.append([])
            greedy_log_probs.append([])
            greedy_log_likelihoods.append([])

        return {
            "greedy_texts": greedy_texts,
            "greedy_log_probs": greedy_log_probs,
            "greedy_log_likelihoods": greedy_log_likelihoods,
            "greedy_tokens": greedy_tokens,
        }


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
        n_alternatives: int = 10,
    ):
        super().__init__()
        self.output_attentions = output_attentions
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
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=self.output_attentions,
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
            if self.output_attentions:
                attentions = out.attentions
            embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type
            )

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
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:text_length]))
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

        attention_all = []
        if self.output_attentions:
            for i in range(len(texts)):
                c = len(cut_sequences[i])
                attn_mask = np.zeros(
                    shape=(
                        model.model.config.num_attention_heads
                        * model.model.config.num_hidden_layers,
                        c,
                        c,
                    )
                )
                for j in range(1, c):
                    attn_mask[:, j, :j] = (
                        torch.vstack(
                            [
                                attentions[j][layer][0][head][0][-j:]
                                for layer in range(len(attentions[j]))
                                for head in range(len(attentions[j][layer][0]))
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                attention_all.append(attn_mask.max(0))

        if model.model_type == "CausalLM":
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
