import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class EnergyCalculator(StatCalculator):
    """
    Computes the energy statistics required by the Spilled Energy method
    (Minut, Dewidar & Masi, ICLR 2026) from the model's **raw logits**.

    Why this calculator has to exist
    --------------------------------
    ``GreedyProbsCalculator`` stores ``greedy_log_probs``, but those are
    ``log_softmax(logits)``: ``WhiteboxModel.generate`` installs a
    ``_ScoresProcessor`` that overwrites ``out.scores`` with
    ``scores.log_softmax(-1)``. Since ``log_softmax(theta) = theta - Z`` with
    ``Z = logsumexp_k theta[k]``, the log-partition ``Z`` is normalised away
    (``logsumexp`` of any log-prob row is exactly 0) and is **not recoverable**
    from any existing statistic. The Spilled Energy scores need ``Z``, so we
    re-run the model teacher-forced over ``prompt + greedy generation`` and read
    the raw logits before any normalisation.

    Statistics produced (per sample, generation of length ``N``)
    ------------------------------------------------------------
    * ``energy_token_logits`` : ``float32[N]`` -- raw logit of the *sampled*
      token at each generation step, ``theta_j[id(x_j)]``. The logit energy of
      the paper is ``E^l_j = -energy_token_logits[j]``.
    * ``energy_lse`` : ``float32[N + 1]`` -- log-partition
      ``Z_j = logsumexp_k theta_j[k]`` at each generation step, **plus one extra
      entry** for the step immediately after the last generated token. The
      marginal energy is ``E^m_j = -energy_lse[j]``. The trailing entry is what
      makes the *adjacent-step* spilled energy defined for the final token:
      ``dE_j = Z_{j+1} - theta_j[id(x_j)]``.

    Only these scalars are kept -- the ``[N, V]`` logit matrix is reduced on the
    fly and never stored (a 3B model with ``V ~ 152k`` would otherwise cost
    ~12 MB per sample).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return (
            ["energy_token_logits", "energy_lse", "energy_trailing_terminators"],
            ["greedy_tokens"],
        )

    def __init__(
        self,
        batch_chunk_size: int = 0,
        fp32_projection: bool = True,
        vocab_chunk: int = 32768,
    ):
        """
        Parameters:
            batch_chunk_size (int): if > 0, run the teacher-forced pass in chunks
                of this many sequences to bound peak memory. 0 means one pass over
                the whole batch.
            fp32_projection (bool): recompute the final vocabulary projection in
                float32 for the handful of rows actually used. ``dE`` is a
                difference of two large, similar quantities and amplifies any
                logit error by roughly 6.5x on Qwen2.5-3B, so the extra precision
                is cheap insurance (~13 MB). Measured effect on the reported runs
                was negligible, which localises the residual fp16 error to the
                transformer body rather than to the lm_head; see the report.
            vocab_chunk (int): vocabulary chunk size for the fp32 projection, to
                avoid materialising a float32 copy of the whole lm_head weight
                (1.2 GB for a 152k vocabulary).
        """
        super().__init__()
        self.batch_chunk_size = batch_chunk_size
        self.fp32_projection = fp32_projection
        self.vocab_chunk = vocab_chunk

    def _project_fp32(self, hidden_rows, lm_head):
        """float32 logits for a few rows, chunked over the vocabulary."""
        w = lm_head.weight
        bias = getattr(lm_head, "bias", None)
        h = hidden_rows.float()
        out = torch.empty(
            (h.shape[0], w.shape[0]), dtype=torch.float32, device=h.device
        )
        for s in range(0, w.shape[0], self.vocab_chunk):
            e = min(s + self.vocab_chunk, w.shape[0])
            out[:, s:e] = h @ w[s:e].float().t()
            if bias is not None:
                out[:, s:e] += bias[s:e].float()
        return out

    @staticmethod
    def _count_trailing_terminators(tokens, tokenizer) -> int:
        """How many tokens at the END of a generation are terminators.

        A terminator here is the EOS/pad id, or a token that decodes to nothing
        but whitespace (the trailing newline that ``stop_strings: ["\\n"]``
        produces). Counted from the end and stopping at the first real token, so
        whitespace inside an answer is never touched.

        Why this matters for pooling: GreedyProbsCalculator trims with
        ``length = j + 1``, so the terminator is INSIDE ``greedy_tokens`` and
        therefore inside the pooling window. TriviaQA answers are short -- a
        median generation is ~4 tokens, of which the newline and the EOS are two.
        Roughly half of the window is then a maximally predictable token whose
        score is near-constant across samples and unrelated to correctness, which
        dilutes mean pooling and can dominate min pooling outright.

        At least one token is always kept, so a degenerate all-whitespace
        generation still yields a score rather than an empty window.
        """
        eos = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
        n = 0
        for tid in reversed(list(tokens)):
            if len(tokens) - n <= 1:
                break
            if tid in eos or tokenizer.decode([tid]).strip() == "":
                n += 1
            else:
                break
        return n

    @staticmethod
    def _assert_finite(tok_logits, lse, rows, sample_idx: int) -> None:
        """Fail loudly on non-finite logits instead of poisoning the scores.

        Spilled Energy reads RAW logits, so an inf or NaN does not get normalised
        away the way it would in a log-softmax: it propagates straight into the
        energies, through pooling, and into PRR as a plausible-looking number.
        A silent NaN here is far worse than a crash.

        The usual cause is fp16 range overflow. Models trained in bf16 (Qwen2.5
        among them) can produce activations outside fp16's much narrower dynamic
        range; on Turing GPUs such as the T4 there is no hardware bf16, so fp16
        is the only half-precision option. The visible symptom downstream is
        argmax collapsing to token id 0.
        """
        bad_tok = ~torch.isfinite(tok_logits)
        bad_lse = ~torch.isfinite(lse)
        if not (bad_tok.any() or bad_lse.any()):
            return

        steps = torch.nonzero(bad_tok | bad_lse[: len(bad_tok)]).flatten().tolist()
        n_bad_rows = int((~torch.isfinite(rows)).any(dim=-1).sum())
        raise RuntimeError(
            f"EnergyCalculator: non-finite logits in sample {sample_idx}.\n"
            f"  non-finite sampled-token logits : {int(bad_tok.sum())}/{len(bad_tok)}\n"
            f"  non-finite log-partitions       : {int(bad_lse.sum())}/{len(bad_lse)}\n"
            f"  decoding steps with any non-finite vocab entry: {n_bad_rows}\n"
            f"  first affected steps            : {steps[:10]}\n"
            f"  dtype                           : {rows.dtype}\n"
            "This is almost always fp16 range overflow (Qwen2.5 was trained in "
            "bf16; a T4 has no hardware bf16). The energies would be garbage, so "
            "the run is aborted rather than producing meaningless PRR. Re-run in "
            "bfloat16 or float32 if the hardware allows it; otherwise keep "
            "attn_implementation='sdpa' and output_attentions=False so that "
            "transformers' left-padding guard stays active."
        )

    def _prompt_ids(self, model: WhiteboxModel, texts: List[str]) -> List[List[int]]:
        """Tokenize each prompt individually (no padding) to get true lengths."""
        tokenizer = model.tokenizer
        ids = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=True, return_attention_mask=False)
            ids.append(list(enc["input_ids"]))
        return ids

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Runs a single teacher-forced forward pass over ``prompt + generation`` and
        reduces the raw logits to the two per-step scalar sequences described above.

        Parameters:
            dependencies (Dict[str, np.ndarray]): must contain 'greedy_tokens'.
            texts (List[str]): input texts batch used for the generation.
            model (WhiteboxModel): the model used for generation.
        Returns:
            Dict[str, np.ndarray]: 'energy_token_logits', 'energy_lse' and
                'energy_trailing_terminators'.
        """
        if getattr(model, "model_type", "CausalLM") != "CausalLM":
            raise NotImplementedError(
                "EnergyCalculator currently supports CausalLM models only; "
                f"got model_type={getattr(model, 'model_type', None)!r}."
            )

        greedy_tokens = dependencies["greedy_tokens"]
        prompt_ids = self._prompt_ids(model, texts)

        tokenizer = model.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        device = model.device()

        # Build right-padded teacher-forcing batch: [prompt || generation].
        # Right padding keeps every real position's causal context pad-free, so
        # per-sample indexing needs only that sample's own prompt length.
        seqs = [list(p) + list(g) for p, g in zip(prompt_ids, greedy_tokens)]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths) if lengths else 0

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        chunk = self.batch_chunk_size if self.batch_chunk_size > 0 else len(seqs)
        token_logits_out: List[np.ndarray] = []
        lse_out: List[np.ndarray] = []
        trailing_out: List[int] = []

        with torch.no_grad():
            for start in range(0, len(seqs), chunk):
                stop = min(start + chunk, len(seqs))
                lm_head = getattr(model.model, "lm_head", None)
                use_fp32 = self.fp32_projection and lm_head is not None
                out = model.model(
                    input_ids=input_ids[start:stop],
                    attention_mask=attention_mask[start:stop],
                    output_hidden_states=use_fp32,
                )
                logits = out.logits  # [b, max_len, V]
                hidden = out.hidden_states[-1] if use_fp32 else None

                for local_i in range(stop - start):
                    i = start + local_i
                    p_len = len(prompt_ids[i])
                    n_gen = len(greedy_tokens[i])
                    if n_gen == 0:
                        token_logits_out.append(np.zeros(0, dtype=np.float32))
                        lse_out.append(np.zeros(0, dtype=np.float32))
                        trailing_out.append(0)
                        continue

                    # Position p_len - 1 + j predicts generated token j.
                    # Take j = 0..n_gen-1 for the sampled-token logits, and
                    # j = 0..n_gen for the log-partitions (one extra step).
                    first = p_len - 1
                    last = p_len - 1 + n_gen  # inclusive -> +1 in the slice
                    if hidden is not None:
                        # Redo the projection in float32 for just these rows: dE is
                        # a cancelling difference and inherits ~6.5x of any logit
                        # error (see __init__).
                        rows = self._project_fp32(
                            hidden[local_i, first : last + 1, :], lm_head
                        )
                    else:
                        rows = logits[
                            local_i, first : last + 1, :
                        ].float()  # [n_gen+1, V]

                    lse = torch.logsumexp(rows, dim=-1)  # [n_gen+1]

                    tok = torch.tensor(
                        list(greedy_tokens[i]), dtype=torch.long, device=rows.device
                    )
                    tok_logits = rows[:n_gen].gather(1, tok.unsqueeze(1)).squeeze(1)

                    self._assert_finite(tok_logits, lse, rows, i)

                    token_logits_out.append(
                        tok_logits.cpu().numpy().astype(np.float32, copy=False)
                    )
                    lse_out.append(lse.cpu().numpy().astype(np.float32, copy=False))
                    trailing_out.append(
                        self._count_trailing_terminators(greedy_tokens[i], tokenizer)
                    )

                del out, logits, hidden

        return {
            "energy_token_logits": token_logits_out,
            "energy_lse": lse_out,
            "energy_trailing_terminators": trailing_out,
        }
