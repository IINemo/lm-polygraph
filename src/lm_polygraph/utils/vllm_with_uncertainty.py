from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from collections import OrderedDict
from typing import Tuple
import numpy as np
import logging
from lm_polygraph.estimators import Estimator
from lm_polygraph.stat_calculators.extract_claims import Claim
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
import torch

# Optional dependency (only needed if output_hidden_states=True)
try:
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator
except Exception:
    VllmHiddenStatesGenerator = None

log = logging.getLogger()


def _safe_float_uncertainty(u: Any) -> float:
    """Robustly extract a scalar from estimator output."""
    if isinstance(u, (float, int)):
        return float(u)
    try:
        arr = np.asarray(u)
        if arr.size == 0:
            return 0.0
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(u[0][0])
        except Exception:
            return float(u[0]) if isinstance(u, (list, tuple)) and len(u) else 0.0


@dataclass
class RequestOutputWithUncertainty:
    """Extends vLLM RequestOutput to include uncertainty scores + optional hidden states payload."""
    request_output: RequestOutput
    uncertainty_scores: List[float]  # One score per output sequence

    def __getattr__(self, name):
        return getattr(self.request_output, name)


class VLLMWithUncertainty:
    def __init__(
            self,
            llm: LLM,
            stat_calculators: List,
            estimator: Estimator,
            n_logprobs: int = 0,
            output_hidden_states: bool = False,
            hs_layer_ids: Optional[List[int]] = None,
            hs_generator_kwargs: Optional[dict] = None,
            prompt_logprobs: bool = False,
            hs_cache_max_seqs: int = 32,
            hs_recompute_on_miss: bool = True,
            hs_strict: bool = False,
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.stat_calculators = stat_calculators
        self.estimator = estimator
        self.n_logprobs = n_logprobs
        self.prompt_logprobs = prompt_logprobs

        self.output_hidden_states = output_hidden_states
        self.hs_layer_ids = hs_layer_ids or []
        self.hs_generator_kwargs = hs_generator_kwargs or {}

        self._hs_gen = None  # lazy init

        self.hs_cache_max_seqs = hs_cache_max_seqs
        self.hs_recompute_on_miss = hs_recompute_on_miss
        self.hs_strict = hs_strict

        self._hs_seq_cache = OrderedDict()

        if self.output_hidden_states:
            if VllmHiddenStatesGenerator is None:
                raise ImportError(
                    "output_hidden_states=True requires "
                    "`speculators.data_generation.vllm_hidden_states_generator.VllmHiddenStatesGenerator` "
                    "(install `speculators`)."
                )
            if not self.hs_layer_ids:
                raise ValueError("output_hidden_states=True requires hs_layer_ids (e.g., [2, 10, 20]).")

    def _get_hs_generator(self) -> "VllmHiddenStatesGenerator":
        if self._hs_gen is not None:
            return self._hs_gen

        # You used model_path=... when constructing the generator.
        # We try to infer model path/name from llm, but often you’ll want to pass it explicitly
        # via hs_generator_kwargs["model_path"].
        if "model_path" not in self.hs_generator_kwargs:
            # Best effort; many vLLM LLM objects store model name under llm.model or llm.llm_engine.model_config.model
            model_path = getattr(self.llm, "model", None)
            if model_path is None:
                raise ValueError(
                    "Please pass hs_generator_kwargs={'model_path': '...'} so the hidden-states generator knows what to load."
                )
            self.hs_generator_kwargs["model_path"] = model_path

        self._hs_gen = VllmHiddenStatesGenerator(
            layer_ids=self.hs_layer_ids,
            **self.hs_generator_kwargs,
        )
        return self._hs_gen

    def _normalize_prompts(self, prompts: Union[str, List[str]]) -> List[str]:
        return [prompts] if isinstance(prompts, str) else list(prompts)

    def _prompts_to_token_ids(self, prompts_list: List[str]) -> List[List[int]]:
        # Match your example (add_special_tokens=True)
        # If your generation uses different prompt formatting, keep this consistent.
        return [self.tokenizer(p, add_special_tokens=True).input_ids for p in prompts_list]

    # VllmHiddenStatesGenerator may reuse KV-cached prefixes and return hidden states
    # only for newly computed suffix tokens.
    # To recover full-sequence hidden states, we cache hidden states from previous
    # batches and stitch cached prefix states with the returned suffix states.
    @staticmethod
    def _flat_hs_layer(t: torch.Tensor) -> torch.Tensor:
        # Normalize shape to (seq_len, hidden_dim), keep on CPU
        return t.reshape(-1, t.shape[-1]).detach().cpu()

    def _payload_seq_len(self, payload: Optional[dict]) -> int:
        if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
            return 0
        t0 = payload["hidden_states"][0]
        return int(t0.reshape(-1, t0.shape[-1]).shape[0])

    def _cache_hs_sequence(self, token_ids: List[int], payload: Optional[dict]) -> None:
        if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
            return
        key = tuple(int(x) for x in token_ids)
        hs_layers = [self._flat_hs_layer(h).clone() for h in payload["hidden_states"]]
        self._hs_seq_cache[key] = hs_layers
        self._hs_seq_cache.move_to_end(key)

        while len(self._hs_seq_cache) > self.hs_cache_max_seqs:
            self._hs_seq_cache.popitem(last=False)

    def _lookup_prefix_hs(self, prefix_ids: List[int]) -> Optional[List[torch.Tensor]]:
        """
        Return hidden states for exactly prefix_ids if available, or from any cached
        full sequence that starts with prefix_ids.
        """
        n = len(prefix_ids)
        if n == 0:
            return []

        key = tuple(int(x) for x in prefix_ids)

        # exact match
        exact = self._hs_seq_cache.get(key)
        if exact is not None:
            return [h[:n].clone() for h in exact]

        # recent-first scan for any super-sequence
        for seq_key, seq_layers in reversed(self._hs_seq_cache.items()):
            if len(seq_key) >= n and seq_key[:n] == key:
                return [h[:n].clone() for h in seq_layers]

        return None

    def _reset_hs_prefix_cache(self, hs_gen) -> bool:
        """
        Best-effort reset of prefix cache ONLY for the hidden-states generator path.
        """
        try:
            sch = getattr(hs_gen, "scheduler", None)
            if sch is None:
                return False

            # Some versions expose reset on scheduler
            if hasattr(sch, "reset_prefix_cache"):
                return bool(sch.reset_prefix_cache())

            # Others expose it on kv_cache_manager
            kvm = getattr(sch, "kv_cache_manager", None)
            if kvm is not None and hasattr(kvm, "reset_prefix_cache"):
                return bool(kvm.reset_prefix_cache())
        except Exception as e:
            log.warning("Could not reset HS generator prefix cache: %s", e)

        return False

    def _stitch_payload_from_cache(
            self,
            req_ids: List[int],
            payload: Optional[dict],
    ) -> Optional[dict]:
        """
        If payload contains only uncached suffix hidden states, stitch the missing prefix
        from self._hs_seq_cache.
        """
        if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
            return payload

        expected = len(req_ids)
        hs_suffix = [self._flat_hs_layer(h) for h in payload["hidden_states"]]
        got = hs_suffix[0].shape[0]

        if got == expected:
            out = dict(payload)
            out["hidden_states"] = hs_suffix
            return out

        if got > expected:
            out = dict(payload)
            out["hidden_states"] = [h[:expected].contiguous() for h in hs_suffix]
            return out

        # got < expected -> missing cached prefix part
        missing = expected - got
        prefix_ids = req_ids[:missing]
        prefix_hs = self._lookup_prefix_hs(prefix_ids)
        if prefix_hs is None:
            log.info(
                "HS stitch: cache miss for prefix (missing=%d, suffix=%d, expected=%d).",
                missing, got, expected
            )
            return None

        full = [torch.cat([p, s], dim=0) for p, s in zip(prefix_hs, hs_suffix)]
        out = dict(payload)
        out["hidden_states"] = full
        log.info(
            "HS stitch: stitched %d tokens from cache + %d suffix tokens = %d total.",
            missing, got, expected
        )
        return out

    def _gen_hidden_states(self, full_token_ids):
        """
        Try normal generation first.
        If got_len < expected_len (APC reused prefix), stitch missing prefix from local cache.
        If still unresolved, recompute unresolved sequences after resetting HS prefix cache.
        """
        hs_gen = self._get_hs_generator()
        raw_payload = hs_gen.generate(full_token_ids)

        payload: List[Optional[dict]] = [None] * len(full_token_ids)
        unresolved: List[int] = []

        # Pass 1: stitch from local cache
        for i, req_ids in enumerate(full_token_ids):
            pay = raw_payload[i] if raw_payload is not None and i < len(raw_payload) else None
            stitched = self._stitch_payload_from_cache(req_ids, pay)

            if stitched is None:
                unresolved.append(i)
                payload[i] = pay
                continue

            got_len = self._payload_seq_len(stitched)
            if got_len >= len(req_ids):
                self._cache_hs_sequence(req_ids, stitched)
            else:
                unresolved.append(i)

            payload[i] = stitched

        # Pass 2: targeted recompute for unresolved
        if unresolved and self.hs_recompute_on_miss:
            log.warning(
                "HS length mismatch for %d/%d sequences; recomputing unresolved with HS cache reset.",
                len(unresolved), len(full_token_ids),
            )

            for i in unresolved:
                req_ids = full_token_ids[i]

                # Reset only HS path cache, not main LLM engine behavior
                self._reset_hs_prefix_cache(hs_gen)

                # Recompute single sequence (avoid batch-level cross-contamination)
                one = hs_gen.generate([req_ids])
                one_pay = one[0] if one else None
                stitched = self._stitch_payload_from_cache(req_ids, one_pay)

                got = self._payload_seq_len(stitched) if stitched is not None else 0
                if got >= len(req_ids):
                    payload[i] = stitched
                    self._cache_hs_sequence(req_ids, stitched)
                    continue

                msg = (
                    f"Could not recover full hidden states for sequence {i}: "
                    f"got_len={got}, expected_len={len(req_ids)}"
                )
                if self.hs_strict:
                    raise RuntimeError(msg)
                log.warning(msg)
                payload[i] = stitched if stitched is not None else one_pay

        return payload

    def _build_flat_full_token_ids(
            self,
            prompts_list: List[str],
            outputs: List[RequestOutput],
    ):
        """
        Returns:
          flat_full_ids: list of full sequences (prompt + generated) for every output candidate
          flat_index: list of (request_i, output_j) matching flat_full_ids
          context_lengths: per request_i prompt length
        """
        prompt_token_ids = self._prompts_to_token_ids(prompts_list)
        context_lengths = [len(ids) for ids in prompt_token_ids]

        flat_full_ids = []
        flat_index = []

        for i, ro in enumerate(outputs):
            p_ids = list(prompt_token_ids[i])
            for j, out in enumerate(ro.outputs):
                gen_ids = list(getattr(out, "token_ids", []) or [])
                flat_full_ids.append(p_ids + gen_ids)
                flat_index.append((i, j))

        return flat_full_ids, flat_index, context_lengths

    def _raw_generate(
            self,
            prompts: Union[str, List[str]],
            sampling_params: Optional[SamplingParams] = None,
    ):

        prompts_list = self._normalize_prompts(prompts)

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Enforce enough logprobs for calculators that need top-N
        sampling_params.logprobs = max(sampling_params.logprobs or 0, self.n_logprobs)
        if sampling_params.logprobs == 0:
            sampling_params.logprobs = 20  # fallback (your previous behavior)

        if self.prompt_logprobs:
            sampling_params.prompt_logprobs = sampling_params.logprobs

        # ---- 1) Generate with vLLM ----
        outputs = self.llm.generate(prompts_list, sampling_params)

        print(f"DEBUG: After generate, output_hidden_states={self.output_hidden_states}")
        import sys
        sys.stdout.flush()

        # ---- 2) Optionally compute hidden states per (prompt, output) ----
        hs_by_req_out: List[List[Optional[dict]]] = [
            [None] * len(ro.outputs) for ro in outputs
        ]
        prompt_token_ids = self._prompts_to_token_ids(prompts_list)
        context_lengths = [len(ids) for ids in prompt_token_ids]

        if self.output_hidden_states:
            # Sleep main LLM to free GPU memory before HS generator loads
            print("\n\n\nSleeping main LLM (level=2) to free GPU memory before HS generator\n\n\n")
            self.llm.sleep(level=2)

            flat_full_ids, flat_index, context_lengths = self._build_flat_full_token_ids(
                prompts_list, outputs
            )

            if flat_full_ids:
                flat_payloads = self._gen_hidden_states(flat_full_ids)
                for k, payload in enumerate(flat_payloads):
                    i, j = flat_index[k]
                    hs_by_req_out[i][j] = payload

        return outputs, hs_by_req_out, context_lengths, prompts_list

    def _construct_request_with_uncertainty(
            self,
            prompts_list: Union[str, List[str]],
            sampling_params: Optional[SamplingParams],
            outputs,
            hs_by_req_out,
            context_lengths,
            compute_uncertainty: bool = True,
    ) -> List[RequestOutputWithUncertainty]:

        results: List[RequestOutputWithUncertainty] = []

        for i, request_output in enumerate(outputs):
            request_scores = []

            if compute_uncertainty:
                for j, out in enumerate(request_output.outputs):
                    out_hs = (
                        hs_by_req_out[i][j]
                        if self.output_hidden_states and j < len(hs_by_req_out[i])
                        else None
                    )

                    # keep your existing pattern
                    out.hidden_states_payload = out_hs

                    # prompt_logprobs can be on request_output depending on vLLM version
                    prompt_lps = getattr(out, "prompt_logprobs", None)
                    if prompt_lps is None:
                        prompt_lps = getattr(request_output, "prompt_logprobs", None)

                    deps = {
                        "vllm_output": out,
                        "token_ids": getattr(out, "token_ids", None),
                        "logprobs": getattr(out, "logprobs", None),
                        "prompt_logprobs": prompt_lps,
                        "context_lengths": [context_lengths[i]],
                    }

                    if out_hs is not None:
                        deps["vllm_hidden_states_output"] = out_hs

                    for calc in self.stat_calculators:
                        deps.update(
                            calc(
                                deps,
                                texts=[prompts_list[i]],
                                model=self,
                                max_new_tokens=getattr(sampling_params, "max_tokens", 0) or 0,
                            )
                        )

                    u = self.estimator(deps)
                    request_scores.append(_safe_float_uncertainty(u))

                    # cache deps
                    request_output.outputs[j].deps = deps

            results.append(
                RequestOutputWithUncertainty(
                    request_output=request_output,
                    uncertainty_scores=request_scores,
                )
            )
        return results

    def generate(
            self,
            prompts: Union[str, List[str]],
            sampling_params: Optional[SamplingParams] = None,
            compute_uncertainty: bool = True,
    ) -> List[RequestOutputWithUncertainty]:
        outputs, hs_by_req_out, context_lengths, prompts_list = self._raw_generate(
            prompts, sampling_params,
        )
        results = self._construct_request_with_uncertainty(
            prompts_list, sampling_params, outputs,
            hs_by_req_out, context_lengths, compute_uncertainty,
        )

        return results

    def _score_from_deps(self, deps: Dict) -> float:
        for calc in self.stat_calculators:
            deps.update(calc(deps, texts=[None], model=self))

        uncertainty = self.estimator(deps)
        if isinstance(uncertainty, list):
            assert len(uncertainty) == 1
            if isinstance(uncertainty[0], list):
                assert len(uncertainty[0]) == 1
                return float(uncertainty[0][0])
            return float(uncertainty[0])
        return float(uncertainty)

    def score(
            self,
            token_ids: List[int],
            logprobs: List[Dict],
            output=None,
            claim_range=None,
    ) -> float:
        """
        Score arbitrary (possibly truncated) sequences.
        """
        if not token_ids or not logprobs:
            return 0.0

        if output is not None and hasattr(output, "deps"):
            deps = output.deps
            deps["claims"] = [[Claim(
                None, None,
                list(range(claim_range[0], claim_range[1])),
            )]]
            log.debug(f"Recovered deps: {deps.keys()}")
        else:
            deps = {
                "token_ids": token_ids,
                "logprobs": logprobs,
            }
        return self._score_from_deps(deps)

    def get_tokenizer(self):
        return self.tokenizer

    def __getattr__(self, name):
        return getattr(self.llm, name)
