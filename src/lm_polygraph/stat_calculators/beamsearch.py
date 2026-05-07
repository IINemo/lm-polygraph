import torch
import gc
import numpy as np
from typing import Dict, List, Tuple, Union
import logging

from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM
from lm_polygraph.stat_calculators import StatCalculator
from .embeddings import get_embeddings_from_output
from .normalize_texts import normalize

log = logging.getLogger("lm_polygraph")


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


class BeamSearchGenerationCalculator(StatCalculator):
    """
    For Whitebox model, calculates beam search generations:
    * tokens of the beam outputs
    * texts of the beam outputs
    * log-likelihoods of the beam outputs
    * embeddings of the last token from mid-layer
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "beamsearch_tokens",
            "beamsearch_texts",
            "beamsearch_log_likelihoods",
            "beamsearch_log_probs",
            "beamsearch_embeddings",
        ], []

    def __init__(self, beams_n: int = 10, num_beam_groups: int | None = None, diversity_penalty: float | None = None):
        super().__init__()
        self.beams_n = beams_n
        self.num_beam_groups = num_beam_groups
        if num_beam_groups == 1:
            self.num_beam_groups = None
        self.diversity_penalty = diversity_penalty

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:

        batch = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        batch_size = len(texts)

        args = {}
        if self.num_beam_groups is not None:
            args["num_beam_groups"] = self.num_beam_groups
        if self.diversity_penalty is not None:
            args["diversity_penalty"] = self.diversity_penalty

        log.info(f'Running BeamSearch with {args}')
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,  # <- Enable embeddings
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                do_sample=False,
                num_beams=self.beams_n,
                num_return_sequences=self.beams_n,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
                **args,
            )

        sequences = out.sequences.cpu()

        beam_tokens = [[] for _ in range(batch_size)]
        beam_texts = [[] for _ in range(batch_size)]
        beam_lls = [[] for _ in range(batch_size)]
        beam_lps = [[] for _ in range(batch_size)]
        beam_embeddings = [[] for _ in range(batch_size)]

        out_wrapper = OutputWrapper()
        if model.model_type == "CausalLM":
            out_wrapper.hidden_states = out.hidden_states
        elif model.model_type == "Seq2SeqLM":
            out_wrapper.encoder_hidden_states = out.encoder_hidden_states
            out_wrapper.decoder_hidden_states = out.decoder_hidden_states

        _, token_embeddings = get_embeddings_from_output(
            out_wrapper,
            batch,
            model.model_type,
            level="token",
            hidden_layer=int(model.model.config.num_hidden_layers // 2),
        )
        del out_wrapper
        del out
        batch = {k: v.cpu() for k, v in batch.items()}
        gc.collect()
        torch.cuda.empty_cache()

        if token_embeddings.dtype == torch.bfloat16:
            token_embeddings = token_embeddings.to(torch.float16)

        with torch.no_grad():
            out_call = model(input_ids=sequences.to(model.device()), output_attentions=False)
            logits = out_call.logits.cpu()
            logits = logits[:, batch['input_ids'].shape[-1] - 1:, :].log_softmax(-1)
            del out_call

        for i in range(batch_size):
            for j in range(self.beams_n):
                index = i * self.beams_n + j
                seq = sequences[index]

                input_len = len(batch["input_ids"][i]) if model.model_type == "CausalLM" else 0
                seq = seq[input_len:]

                toks = []
                ll = []
                for k, token_id in enumerate(seq):
                    token_id = token_id.item()
                    toks.append(token_id)
                    ll.append(logits[index][k][token_id].item())
                    if token_id == model.tokenizer.eos_token_id:
                        break

                beam_tokens[i].append(toks)
                beam_texts[i].append(normalize(model.tokenizer.decode(toks)))
                beam_lls[i].append(ll)
                beam_lps[i].append(sum(ll))

                if len(token_embeddings.shape) > 2:
                    emb = token_embeddings[index, -1].cpu().detach().numpy()
                else:
                    emb = token_embeddings[index].cpu().detach().numpy()
                beam_embeddings[i].append(emb)

        return {
            "beamsearch_tokens": beam_tokens,
            "beamsearch_texts": beam_texts,
            "beamsearch_log_likelihoods": beam_lls,
            "beamsearch_log_probs": beam_lps,
            "beamsearch_embeddings": beam_embeddings,
        }
