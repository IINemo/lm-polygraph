import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from .embeddings import get_embeddings_from_output
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


def _gen_samples(n_samples, model, batch, **kwargs):
    batch_size = len(batch["input_ids"])
    logits, sequences, embeddings = (
        [[] for _ in range(batch_size)],
        [[] for _ in range(batch_size)],
        [],
    )
    with torch.no_grad():
        for k in range(n_samples):
            out = model.generate(**batch, **kwargs)
            cur_logits = torch.stack(out.scores, dim=1)
            embeddings.append(
                {
                    "sample_embeddings_all_decoder": out.hidden_states,
                }
            )
            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    return sequences, sum(logits, []), embeddings


class SamplingGenerationVisualCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.VisualWhiteboxModel), at input texts batch calculates:
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
            "sample_log_probs",
            "sample_tokens",
            "sample_texts",
            "sample_log_likelihoods",
            "sample_embeddings",
        ], []

    def __init__(self, samples_n: int = 10):
        super().__init__()
        self.samples_n = samples_n

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
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
                - 'sample_embeddings' (List[List[List[float]]]): embeddings from the middle layer for the last token of the sampling generation.
        """
        batches = {}

        for text, image in zip(texts, model.images):
            batch = model.processor_visual(
                text=str(text),
                images=image,
                return_tensors="pt",
                return_dict=True,
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
        sequences, logits, embeddings = _gen_samples(
            self.samples_n,
            model,
            batch,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            suppress_tokens=(
                []
                if model.generation_parameters.allow_newlines
                else [
                    t
                    for t in range(len(model.processor_visual.tokenizer))
                    if "\n" in model.processor_visual.tokenizer.decode([t])
                ]
            ),
        )

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        log_likelihoods = [[] for _ in range(len(texts))]
        # if model.model_type == "Seq2SeqLM":
        #     sequences = [seq[1:] for seq in sequences]
        for i in range(len(logits)):
            log_prob, ll, toks = 0, [], []
            inp_size = (
                len(batch["input_ids"][int(i / self.samples_n)])
                # if model.model_type == "CausalLM"
                # else 0
            )
            for j in range(len(sequences[i]) - inp_size):
                cur_token = sequences[i][j + inp_size].item()
                log_prob += logits[i][j][cur_token].item()
                if cur_token == model.processor_visual.tokenizer.eos_token_id:
                    break
                ll.append(logits[i][j][cur_token].item())
                toks.append(cur_token)

            log_likelihoods[int(i / self.samples_n)].append(ll)
            log_probs[int(i / self.samples_n)].append(log_prob)
            tokens[int(i / self.samples_n)].append(toks)
            texts[int(i / self.samples_n)].append(
                model.processor_visual.tokenizer.decode(toks)
            )

        out = OutputWrapper()
        batch_size = len(batch["input_ids"])
        embeddings_last_token = [[] for _ in range(batch_size)]

        num_layers = getattr(model.model.config, "num_hidden_layers", 24)

        for sample_embeddings in embeddings:
            out.hidden_states = sample_embeddings["sample_embeddings_all_decoder"]
            _, cur_token_embeddings = get_embeddings_from_output(
                out,
                batch,
                model.model_type,
                level="token",
                hidden_layer=int(num_layers // 2),
            )

        for i in range(batch_size):
            if len(cur_token_embeddings.shape) > 2:
                embeddings_last_token[i].append(
                    cur_token_embeddings[i, -1].cpu().detach().numpy()
                )
            else:
                embeddings_last_token[i].append(
                    cur_token_embeddings[i].cpu().detach().numpy()
                )

        return {
            "sample_log_likelihoods": log_likelihoods,
            "sample_log_probs": log_probs,
            "sample_tokens": tokens,
            "sample_texts": texts,
            "sample_embeddings": embeddings_last_token,
        }
