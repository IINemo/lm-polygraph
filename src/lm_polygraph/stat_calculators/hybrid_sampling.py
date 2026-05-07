# hybrid_beam_then_sampling_calculator.py

import gc
import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from transformers import LogitsProcessor, LogitsProcessorList

from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


def _newline_suppress_list(tokenizer, allow_newlines: bool) -> List[int]:
    if allow_newlines:
        return []
    # Collect single-token IDs that decode to a string containing '\n'
    return [
        t
        for t in range(len(tokenizer))
        if "\n" in tokenizer.decode([t])
    ]


def _stack_scores_to_logprobs(out_scores: List[torch.Tensor], model_type: str) -> List[torch.Tensor]:
    """
    Convert a list of per-step logits (after processors, before softmax) into per-step log-probs.
    Handles vLLM layout if ever needed for consistency (though we don't use vLLM here for sampling).
    """
    cur_logits = torch.stack(out_scores, dim=1)  # [B*, T, V] for HF; [T, B*, V] for some backends
    if model_type == "vLLMCausalLM":
        cur_logits = cur_logits.transpose(1, 0)
    return cur_logits.log_softmax(-1)  # list->tensor of shape [B*, T, V] in log-prob space


class _ExcludeBeamsProcessor(LogitsProcessor):
    """
    For each expanded row, subtracts from next-token prob the *remaining path mass*
    that would complete any precomputed beam whose prefix matches the current prefix.
    Then renormalizes over the vocab.

    beams_info: List[List[dict]] per base example, dict has:
      - "tokens": List[int]              (generated beam tokens incl. EOS if present)
      - "tail_logprobs": List[float]     (tail[j] = sum_{t=j..end} log p(b_t | prefix_t))

    prefix_offsets: List[int]  (prompt len for CausalLM, 1 for Seq2Seq to skip decoder_start)
    expand_factor: int         (num_return_sequences used in sampling call)
    """

    def __init__(self, beams_info: List[List[dict]], prefix_offsets: List[int], expand_factor: int):
        self.beams_info = beams_info
        self.prefix_offsets = prefix_offsets
        self.expand_factor = max(1, int(expand_factor))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        Bcur, V = scores.shape
        device = scores.device
        probs = torch.softmax(scores, dim=-1)  # [Bcur, V]

        for i in range(Bcur):
            base_idx = i // self.expand_factor
            offset = self.prefix_offsets[base_idx]
            prefix = input_ids[i, offset:].tolist()
            j = len(prefix)

            if not self.beams_info[base_idx]:
                continue

            subtract_mass: Dict[int, float] = {}
            for beam in self.beams_info[base_idx]:
                tokens = beam["tokens"]
                if j < len(tokens) and prefix == tokens[:j]:
                    next_tok = tokens[j]
                    # Remaining path mass from here: exp(tail logprob at j)
                    r = math.exp(beam["tail_logprobs"][j])
                    subtract_mass[next_tok] = subtract_mass.get(next_tok, 0.0) + r

            if subtract_mass:
                idxs = torch.tensor(list(subtract_mass.keys()), dtype=torch.long, device=device)
                delta = torch.tensor([subtract_mass[k.item()] for k in idxs], dtype=probs.dtype, device=device)

                probs_i = probs[i]
                probs_i[idxs] = torch.clamp(probs_i[idxs] - delta, min=0.0)

                Z = probs_i.sum()
                if Z.item() > 0:
                    probs[i] = probs_i / Z
                # else: fall back to unmodified probs (rare numeric edge)

        eps = 1e-20
        probs = torch.clamp(probs, min=eps)
        return torch.log(probs)


class HybridBeamThenConditionalSamplingCalculator(StatCalculator):
    """
    First: run beam search to get `num_beams` sequences and their per-token log-probs and embeddings.
    Then: draw (`num_samples - num_beams`) *independent* multinomial samples from the distribution
          conditioned on not producing any of those beam sequences, by subtracting each beam's
          path mass along matching prefixes during sampling.

    Returns combined stats under keys:
      beamsearch{num_beams}sample{num_samples-num_beams}_texts
      beamsearch{num_beams}sample{num_samples-num_beams}_tokens
      beamsearch{num_beams}sample{num_samples-num_beams}_log_likelihoods
      beamsearch{num_beams}sample{num_samples-num_beams}_log_probs
      beamsearch{num_beams}sample{num_samples-num_beams}_embeddings
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        # Keys are dynamic (depend on init parameters). This static stub is for compatibility.
        # The actual __call__ returns keys formatted with the instance's num_beams/num_samples.
        return [], []

    def __init__(self, num_beams: int = 1, num_samples: int = 10):
        super().__init__()
        assert num_beams >= 1, "num_beams must be >= 1"
        assert num_samples > num_beams, "num_samples must be > num_beams"
        self.num_beams = num_beams
        self.num_samples = num_samples

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:

        if getattr(model, "model_type", None) == "vLLMCausalLM":
            # vLLM does not reliably support custom HuggingFace logits processors across adapters.
            # For correctness, we do not attempt the exclusion sampling on vLLM.
            raise NotImplementedError("HybridBeamThenConditionalSamplingCalculator is not supported for vLLM backends.")

        # ====== Tokenize & to device ======
        batch = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        batch_size = len(texts)

        # ====== Common generation args ======
        suppress_tokens = _newline_suppress_list(model.tokenizer, model.generation_parameters.allow_newlines)

        # ====== 1) BEAM SEARCH ======
        with torch.no_grad():
            out_beam = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                do_sample=False,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                suppress_tokens=suppress_tokens,
                output_attentions=False,
            )

        sequences_beam = out_beam.sequences.cpu()  # [B*num_beams, Ttot]
        # Per-step log-probs (post-process, pre-sample)
        logprobs_beam = _stack_scores_to_logprobs(out_beam.scores, model.model_type)  # [B*num_beams, Tgen, V]

        # Prepare containers
        beam_tokens = [[] for _ in range(batch_size)]
        beam_texts = [[] for _ in range(batch_size)]
        beam_lls = [[] for _ in range(batch_size)]
        beam_lps = [[] for _ in range(batch_size)]

        # Iterate beams per example
        for i in range(batch_size):
            prompt_len = len(batch["input_ids"][i]) if model.model_type == "CausalLM" else 0
            for j in range(self.num_beams):
                index = i * self.num_beams + j
                seq_full = sequences_beam[index]
                seq_gen = seq_full[prompt_len:]  # generated portion for CausalLM; whole seq for Seq2Seq

                # Build toks & per-token ll until EOS
                toks, lls = [], []
                for t, tok_id in enumerate(seq_gen):
                    tok = tok_id.item()
                    # per-step logprob under the (processed) distribution
                    if tok == model.tokenizer.eos_token_id:
                        break
                    lls.append(logprobs_beam[index, t, tok].item())
                    toks.append(tok)

                beam_tokens[i].append(toks)
                beam_texts[i].append(model.tokenizer.decode(toks))
                beam_lls[i].append(lls)
                beam_lps[i].append(sum(lls))

        # Build beam path info for exclusion processor
        beams_info: List[List[dict]] = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(self.num_beams):
                lls = beam_lls[i][j]
                toks = beam_tokens[i][j]
                # Tail log-prob at position j is sum(lls[j:])
                tail_logprobs = []
                running = 0.0
                for val in reversed(lls):
                    running += val
                    tail_logprobs.append(running)
                tail_logprobs.reverse()  # now tail_logprobs[j] = sum_{t=j..} lls[t]

                beams_info[i].append({"tokens": toks, "tail_logprobs": tail_logprobs})

        # Prefix offsets used inside logits processor
        if model.model_type == "CausalLM":
            prefix_offsets = [len(batch["input_ids"][i]) for i in range(batch_size)]
        else:
            # For Seq2Seq decoder, the first token is usually BOS/decoder_start; we condition on tokens after that
            prefix_offsets = [1 for _ in range(batch_size)]

        del out_beam
        gc.collect()
        torch.cuda.empty_cache()

        # ====== 2) CONDITIONAL SAMPLING (independent draws) ======
        n_stoch = self.num_samples - self.num_beams

        sample_tokens = [[] for _ in range(batch_size)]
        sample_texts = [[] for _ in range(batch_size)]
        sample_lls = [[] for _ in range(batch_size)]
        sample_lps = [[] for _ in range(batch_size)]

        logits_proc = _ExcludeBeamsProcessor(beams_info, prefix_offsets, expand_factor=n_stoch)

        with torch.no_grad():
            out_s = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                do_sample=True,
                num_beams=1,
                num_return_sequences=n_stoch,
                suppress_tokens=suppress_tokens,
                logits_processor=logits_proc,
            )
        seqs = out_s.sequences.cpu()
        logprobs_s = _stack_scores_to_logprobs(out_s.scores, model.model_type)  # [B, Tgen, V]

        for row in range(batch_size * n_stoch):
            base_idx = row // n_stoch
            prompt_len = len(batch["input_ids"][base_idx]) if model.model_type == "CausalLM" else 0
            seq_gen = seqs[row][prompt_len:]
            toks, lls = [], []
            for t, tok_id in enumerate(seq_gen):
                tok = tok_id.item()
                lls.append(logprobs_s[row, t, tok].item())
                toks.append(tok)
                if tok == model.tokenizer.eos_token_id:
                    break
            sample_tokens[base_idx].append(toks)
            sample_texts[base_idx].append(model.tokenizer.decode(toks))
            sample_lls[base_idx].append(lls)
            sample_lps[base_idx].append(sum(lls))

        # ====== Combine beams + samples under dynamic keys ======
        key_prefix = f"beamsearch{self.num_beams}sample{self.num_samples - self.num_beams}"

        # Merge: beams first, then conditional samples
        all_tokens = [beam_tokens[i] + sample_tokens[i] for i in range(batch_size)]
        all_texts = [beam_texts[i] + sample_texts[i] for i in range(batch_size)]
        all_lls = [beam_lls[i] + sample_lls[i] for i in range(batch_size)]
        all_lps = [beam_lps[i] + sample_lps[i] for i in range(batch_size)]

        # Clean up GPU memory

        return {
            f"{key_prefix}_tokens": all_tokens,
            f"{key_prefix}_texts": all_texts,
            f"{key_prefix}_log_likelihoods": all_lls,
            f"{key_prefix}_log_probs": all_lps,
        }
