import torch
import numpy as np

from typing import Dict, List, Tuple

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class GreedyProbsVisualCalculator(StatCalculator):
    """
    For WhiteboxModelvLLM model, at input texts batch calculates:
    * sampled texts
    * tokens of the sampled texts
    * probabilities of the sampled tokens generation
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
        samples_n: int = 10,
    ):
        super().__init__()
        self.samples_n = samples_n
        self.output_attentions = output_attentions
        self.n_alternatives = n_alternatives

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

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
        batches = {}

        for text, image in zip(texts, model.images):
            batch = model.processor_visual(
                text=str(text),
                images=image,
                return_tensors="pt",
            )
            batch = {k: v.to(model.device()) for k, v in batch.items()}
            if not batches:
                batches = {k: [v] for k, v in batch.items()}
            else:
                for key in batch:
                    batches[key].append(batch[key])
        batch: Dict[str, torch.Tensor] = {
            key: torch.cat(value, dim=0) for key, value in batches.items()
        }

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=self.output_attentions,
            )
            logits = torch.stack(out.scores, dim=1)
            sequences = out.sequences
            if self.output_attentions:
                attentions = out.attentions
            embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type
            )[1]

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            seq = sequences[i, -logits.shape[1] :].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.processor_visual.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.processor_visual.tokenizer.decode(seq[:text_length]))
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
                    key=lambda x: x[0] == cut_sequences[-1][j], reverse=True
                )

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        attention_all = []
        if self.output_attentions:
            num_heads = getattr(
                model.model.config, "num_attention_heads", 32
            )  # Default to 32
            num_layers = getattr(
                model.model.config, "num_hidden_layers", 24
            )  # Default to 24
            for i in range(len(texts)):
                c = len(cut_sequences[i])
                attn_mask = np.zeros((num_heads * num_layers, c, c))

                for j in range(1, c):
                    stacked_attn = (
                        torch.vstack(
                            [
                                attentions[j][layer][0][head][-j:].reshape(-1, 1)
                                for layer in range(len(attentions[j]))
                                for head in range(len(attentions[j][layer][0]))
                            ]
                        )
                        .cpu()
                        .numpy()
                    )

                    expected_shape = attn_mask[:, j, :j].shape
                    actual_shape = stacked_attn.shape

                    if actual_shape == expected_shape:
                        attn_mask[:, j, :j] = stacked_attn
                    else:
                        attn_mask[:, j, :j] = stacked_attn[
                            : expected_shape[0], : expected_shape[1]
                        ]

                attention_all.append(attn_mask.max(0))

        embeddings_dict = {
            "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
        }

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }
        if self.output_attentions:
            result_dict.update({"attention_all": attention_all})
            result_dict.update({"tokenizer": model.processor_visual.tokenizer})
        result_dict.update(embeddings_dict)
        return result_dict
