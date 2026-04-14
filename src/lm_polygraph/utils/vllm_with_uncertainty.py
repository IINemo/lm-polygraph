"""
vLLM model wrapper with uncertainty estimation and optional hidden states extraction.

Wraps a vLLM LLM instance with lm-polygraph uncertainty scoring.
Supports generation with immediate scoring, standalone scoring of pre-extracted
logprobs, and optional hidden states capture for UQ head-based estimators.

Usage:
    from vllm import LLM
    from lm_polygraph.estimators import MeanTokenEntropy
    from lm_polygraph.stat_calculators import VLLMLogprobsExtractionCalculator, EntropyCalculator
    from lm_polygraph.utils import VLLMWithUncertainty

    llm = LLM(model="Qwen/Qwen2.5-7B")
    model = VLLMWithUncertainty(
        llm=llm,
        stat_calculators=[VLLMLogprobsExtractionCalculator(), EntropyCalculator()],
        estimator=MeanTokenEntropy(),
    )

    # Generate with uncertainty scoring
    results = model.generate(prompts=["What is 2+2?"], sampling_params=params)
    # results[i].uncertainty_scores  -- one score per output sequence

    # Score pre-extracted logprobs
    uncertainty = model.score(token_ids, logprobs)
"""

import logging
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from lm_polygraph.estimators import Estimator
from lm_polygraph.stat_calculators.extract_claims import Claim

# Optional dependency (only needed for Path 1: VllmHiddenStatesGenerator)
try:
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )
except Exception:
    VllmHiddenStatesGenerator = None


log = logging.getLogger()


def _fill_prefix_gaps(
    captured: Dict[str, np.ndarray],
    metadata: Dict[str, Dict],
    prompt_groups: Dict[str, List[str]],
    prompt_tokens: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Fill prefix-cache gaps in captured hidden states.

    Phase 1 — Within-group: for identical prompts, copy the missing prefix
    from the group donor (request with the most prefill tokens).

    Phase 2 — Cross-group (requires prompt_tokens): vLLM caches the shared
    chat-template prefix across different prompts. Uses token-level
    longest-common-prefix (LCP) with a global donor to fill remaining gaps.

    Args:
        captured:      {req_id: numpy_array} — deserialized hidden states
        metadata:      {req_id: {"total_computed": int, "prefill_tokens": int}}
        prompt_groups: {group: [req_id, ...]} — requests with identical prompts
        prompt_tokens: {req_id: [token_ids]} — for cross-group filling
    Returns:
        {req_id: numpy_array} with gaps filled.
    """
    result = dict(captured)

    # Track effective prefill per request (updated as we fill)
    eff_prefill: Dict[str, int] = {
        rid: metadata.get(rid, {}).get("prefill_tokens", 0) for rid in result
    }

    # Phase 1: within-group fill (identical prompts)
    for _, req_ids in prompt_groups.items():
        group_reqs = [rid for rid in req_ids if rid in result]
        if len(group_reqs) < 2:
            continue
        donor_id = max(group_reqs, key=lambda rid: eff_prefill.get(rid, 0))
        donor_prefill = eff_prefill[donor_id]
        donor_arr = result[donor_id]
        for req_id in group_reqs:
            if req_id == donor_id:
                continue
            gap = donor_prefill - eff_prefill[req_id]
            if gap > 0:
                result[req_id] = np.concatenate(
                    [donor_arr[:gap], result[req_id]], axis=0
                )
                eff_prefill[req_id] = donor_prefill

    # Phase 2: cross-group fill (shared token prefix)
    if not prompt_tokens:
        return result

    all_req_ids = list(result.keys())
    if len(all_req_ids) < 2:
        return result

    # Global donor: pick from requests with full prefill, then longest prompt
    full_prefill_reqs = [
        rid
        for rid in all_req_ids
        if eff_prefill.get(rid, 0) >= len(prompt_tokens.get(rid, []))
    ]
    if not full_prefill_reqs:
        return result

    global_donor = max(
        full_prefill_reqs, key=lambda rid: len(prompt_tokens.get(rid, []))
    )
    gd_prefill = eff_prefill[global_donor]
    gd_tokens = prompt_tokens.get(global_donor, [])
    gd_arr = result[global_donor]

    for req_id in all_req_ids:
        if req_id == global_donor:
            continue
        req_toks = prompt_tokens.get(req_id, [])
        req_prompt_len = len(req_toks)
        cur_prefill = eff_prefill[req_id]
        if cur_prefill >= req_prompt_len:
            continue

        # Longest common prefix with global donor
        lcp = 0
        for a, b in zip(gd_tokens, req_toks):
            if a == b:
                lcp += 1
            else:
                break

        missing = req_prompt_len - cur_prefill
        fillable = min(missing, lcp, gd_prefill)
        if fillable > 0:
            result[req_id] = np.concatenate([gd_arr[:fillable], result[req_id]], axis=0)
            eff_prefill[req_id] = cur_prefill + fillable

    return result


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
    """
    vLLM wrapper with uncertainty estimation and optional hidden states extraction.

    Hidden states can be computed using two paths:
    - Path 1 (use_native_hs_capture=False): Uses VllmHiddenStatesGenerator from speculators.
      Requires separate generation pass; main LLM is slept during HS generation.
    - Path 2 (use_native_hs_capture=True): Uses native vLLM capture via collective_rpc.
      Captures HS during main generation; requires vLLM with HS capture extension.

    Args:
        llm: vLLM LLM engine instance.
        stat_calculators: List of lm-polygraph stat calculators to compute statistics
            from generation output (e.g., VLLMLogprobsExtractionCalculator, EntropyCalculator).
        estimator: lm-polygraph Estimator to compute final uncertainty score
            from calculator statistics (e.g., MeanTokenEntropy, Perplexity).
        n_logprobs: Number of top logprobs to request from vLLM per token. Default: 0.
        output_hidden_states: If True, extract hidden states during generation.
            Requires hs_layer_ids to be set. Default: False.
        hs_layer_ids: List of transformer layer indices to capture hidden states from
            (e.g., [2, 10, 20]). Required when output_hidden_states=True.
        prompt_logprobs: If True, request logprobs for prompt tokens. Default: False.
        hs_cache_max_seqs: Max number of sequences to cache hidden states for. Default: 32.
        hs_recompute_on_miss: If True, recompute hidden states on cache miss. Default: True.
        hs_strict: If True, raise on hidden states mismatches. Default: False.
        hs_generator_kwargs: Additional kwargs for VllmHiddenStatesGenerator (Path 1 only).
        use_native_hs_capture: If True, use Path 2 (native capture via collective_rpc).
            Requires enforce_eager=True and HookHiddenStatesExtension worker.
            If False, use Path 1 (VllmHiddenStatesGenerator). Default: False.
    """

    def __init__(
        self,
        llm: LLM,
        stat_calculators: List,
        estimator: Estimator,
        n_logprobs: int = 0,
        output_hidden_states: bool = False,
        hs_layer_ids: Optional[List[int]] = None,
        prompt_logprobs: bool = False,
        hs_cache_max_seqs: int = 32,
        hs_recompute_on_miss: bool = True,
        hs_strict: bool = False,
        hs_generator_kwargs: Optional[dict] = None,
        use_native_hs_capture: bool = False,
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
        self.use_native_hs_capture = use_native_hs_capture

        # Path 2: Native capture variables
        self._hs_extension_ready = False
        self._engine_core = None

        # Multi-step HS accumulator: stores HS from previous generation steps
        # so that prefix-cached tokens can be filled from prior captures.
        # Structure: list of {
        #   "text": str,  — prompt text (string prefix matching)
        #   "layers": {layer_id: np.ndarray}  — HS arrays per layer
        # }
        self._hs_step_cache: List[Dict] = []

        # Path 1: VllmHiddenStatesGenerator variables
        self._hs_gen = None  # lazy init

        self.hs_cache_max_seqs = hs_cache_max_seqs
        self.hs_recompute_on_miss = hs_recompute_on_miss
        self.hs_strict = hs_strict

        self._hs_seq_cache = OrderedDict()

        if self.output_hidden_states:
            if not self.hs_layer_ids:
                raise ValueError(
                    "output_hidden_states=True requires hs_layer_ids (e.g., [2, 10, 20])."
                )

            if self.use_native_hs_capture:
                log.warning(
                    "Native HS capture (use_native_hs_capture=True) uses enforce_eager=True, "
                    "which is 10-20%% slower than CUDA graph mode."
                )
                # Verify vLLM is initialized correctly for native HS capture
                llm_engine = getattr(self.llm, "llm_engine", None)
                if llm_engine is not None:
                    model_config = getattr(llm_engine, "model_config", None)

                    if model_config is not None:
                        # Check enforce_eager=True
                        enforce_eager = getattr(model_config, "enforce_eager", None)
                        if enforce_eager is not True:
                            raise ValueError(
                                f"Native HS capture (use_native_hs_capture=True) requires "
                                f"enforce_eager=True in LLM initialization. Current value: {enforce_eager}. "
                                f"Please initialize LLM with: LLM(..., enforce_eager=True, ...)"
                            )

                        # Check worker_extension_cls contains HookHiddenStatesExtension
                        vllm_config = getattr(llm_engine, "vllm_config", None)
                        parallel_config = getattr(vllm_config, "parallel_config", None)
                        worker_extension_cls = getattr(
                            parallel_config, "worker_extension_cls", None
                        )
                        if worker_extension_cls is None:
                            raise ValueError(
                                "Native HS capture (use_native_hs_capture=True) requires "
                                "worker_extension_cls='hook_hs_extension.HookHiddenStatesExtension' "
                                "in LLM initialization. "
                                "Please initialize LLM with: LLM(..., worker_extension_cls='hook_hs_extension.HookHiddenStatesExtension', ...)"
                            )
                        elif "HookHiddenStatesExtension" not in str(
                            worker_extension_cls
                        ):
                            raise ValueError(
                                f"Native HS capture (use_native_hs_capture=True) requires "
                                f"worker_extension_cls='hook_hs_extension.HookHiddenStatesExtension'. "
                                f"Current value: {worker_extension_cls}. "
                                f"Please initialize LLM with: LLM(..., worker_extension_cls='hook_hs_extension.HookHiddenStatesExtension', ...)"
                            )
            elif VllmHiddenStatesGenerator is None:
                raise ImportError(
                    "output_hidden_states=True with use_native_hs_capture=False requires "
                    "`speculators.data_generation.vllm_hidden_states_generator.VllmHiddenStatesGenerator` "
                    "(install `speculators`)."
                )

    def _ensure_hs_extension(self) -> None:
        if self._hs_extension_ready:
            return
        engine_core = self.llm.llm_engine.engine_core

        # Check initialization result
        setup_result = engine_core.collective_rpc(
            "_setup_hidden_states_capture",
            args=(self.hs_layer_ids,),
        )
        log.info("HS extension setup result: %s", setup_result)

        self._engine_core = engine_core
        self._hs_extension_ready = True
        log.info(
            "HS extension ready (Path 2: native capture). layer_ids=%s",
            self.hs_layer_ids,
        )

    # ------------------------------------------------------------------ #
    # Multi-step HS accumulator (cross-step prefix fill)                 #
    # ------------------------------------------------------------------ #

    def reset_hs_step_cache(self, reset_prefix_cache: bool = True) -> None:
        """Clear the multi-step HS accumulator and (optionally) vLLM APC.

        Call this before starting a new multi-step loop (beam search,
        online BoN).  Resetting the vLLM prefix cache ensures the first
        step captures full hidden states for the prompt — otherwise
        APC may have cached the shared system-prompt prefix from a
        prior call, leaving a permanent gap in every subsequent step.

        Args:
            reset_prefix_cache: If True (default), also reset the vLLM
                automatic prefix cache so the first generation step
                computes hidden states for all prompt tokens.
        """
        self._hs_step_cache = []
        if reset_prefix_cache and self.use_native_hs_capture:
            self._reset_hs_prefix_cache()

    def _find_cached_hs(self, prompt_text: str) -> Optional[Dict[int, np.ndarray]]:
        """Find the longest cached entry whose text is a prefix of *prompt_text*.

        Uses raw prompt strings as keys — immune to BPE re-tokenisation
        differences that break token-level prefix matching.

        Returns {layer_id: hs_array} or None.
        """
        best: Optional[Dict] = None
        best_len = 0
        for entry in self._hs_step_cache:
            key = entry["text"]
            klen = len(key)
            if klen <= best_len or klen > len(prompt_text):
                continue
            if prompt_text[:klen] == key:
                best = entry
                best_len = klen
        if best is not None:
            return best["layers"]
        return None

    def _update_hs_step_cache(
        self, prompt_text: str, layer_id: int, hs_array: np.ndarray
    ) -> None:
        """Store (or extend) HS for a prompt text in the step cache."""
        for entry in self._hs_step_cache:
            if entry["text"] == prompt_text:
                entry["layers"][layer_id] = hs_array
                return
        self._hs_step_cache.append(
            {"text": prompt_text, "layers": {layer_id: hs_array}}
        )

    def _cleanup_hs_step_cache(self, active_texts: List[str]) -> None:
        """Remove cached entries that are not a prefix of any active text."""
        keep = []
        for entry in self._hs_step_cache:
            key = entry["text"]
            if any(a.startswith(key) for a in active_texts):
                keep.append(entry)
        self._hs_step_cache = keep

    def _get_hs_generator(self) -> "VllmHiddenStatesGenerator":
        """Path 1: Get or create VllmHiddenStatesGenerator instance."""
        if self._hs_gen is not None:
            return self._hs_gen

        # You used model_path=... when constructing the generator.
        # We try to infer model path/name from llm, but often you'll want to pass it explicitly
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
        log.info(
            "HS generator ready (Path 1: VllmHiddenStatesGenerator). layer_ids=%s",
            self.hs_layer_ids,
        )
        return self._hs_gen

    def _normalize_prompts(self, prompts: Union[str, List[str]]) -> List[str]:
        return [prompts] if isinstance(prompts, str) else list(prompts)

    def _prompts_to_token_ids(self, prompts_list: List[str]) -> List[List[int]]:
        # Match your example (add_special_tokens=True)
        # If your generation uses different prompt formatting, keep this consistent.
        return [
            self.tokenizer(p, add_special_tokens=True).input_ids for p in prompts_list
        ]

    # VllmHiddenStatesGenerator may reuse KV-cached prefixes and return hidden states
    # only for newly computed suffix tokens.
    # To recover full-sequence hidden states, we cache hidden states from previous
    # batches and stitch cached prefix states with the returned suffix states.
    @staticmethod
    def _flat_hs_layer(t: torch.Tensor) -> torch.Tensor:
        # Normalize shape to (seq_len, hidden_dim), keep on CPU
        return t.reshape(-1, t.shape[-1]).detach().cpu()

    def _payload_seq_len(self, payload: Optional[dict]) -> int:
        if (
            payload is None
            or "hidden_states" not in payload
            or not payload["hidden_states"]
        ):
            return 0
        t0 = payload["hidden_states"][0]
        return int(t0.reshape(-1, t0.shape[-1]).shape[0])

    def _cache_hs_sequence(self, token_ids: List[int], payload: Optional[dict]) -> None:
        if (
            payload is None
            or "hidden_states" not in payload
            or not payload["hidden_states"]
        ):
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

    def _reset_hs_prefix_cache(self, hs_gen=None) -> bool:
        """
        Reset prefix cache for the active hidden states path.
        For Path 2 (native): uses engine_core
        For Path 1 (VllmHiddenStatesGenerator): uses hs_gen parameter
        """
        if self.use_native_hs_capture:
            # Path 2: Native capture
            try:
                self.llm.llm_engine.reset_prefix_cache()
                # self._engine_core.call_utility("reset_prefix_cache")
                return True
            except Exception as e:
                log.warning("Could not reset native prefix cache: %s", e)
                return False
        else:
            # Path 1: VllmHiddenStatesGenerator
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

    def _reset_llm_prefix_cache(self, llm) -> bool:
        """
        Best-effort reset of prefix cache for the main LLM to force full forward pass.

        This invalidates KV cache so that the additional forward pass for HS capture
        processes the entire input sequence (prompt + generated tokens) instead of
        using cached KV pairs from previous generation.
        """
        try:
            # Try to get scheduler from llm
            sch = getattr(llm, "scheduler", None)
            if sch is None:
                # Try llm.llm_engine.scheduler for vLLM
                sch = getattr(llm, "llm_engine", None)
                if sch is not None:
                    sch = getattr(sch, "scheduler", None)

            if sch is None:
                log.warning("Could not find scheduler in LLM for cache reset")
                return False

            # Some versions expose reset on scheduler
            if hasattr(sch, "reset_prefix_cache"):
                return bool(sch.reset_prefix_cache())

            # Others expose it on kv_cache_manager
            kvm = getattr(sch, "kv_cache_manager", None)
            if kvm is not None and hasattr(kvm, "reset_prefix_cache"):
                return bool(kvm.reset_prefix_cache())

            # Try alternative methods
            if hasattr(sch, "reset"):
                return bool(sch.reset())

            if hasattr(sch, "clear"):
                return bool(sch.clear())

            log.warning("Found scheduler but no reset method available")
            return False

        except Exception as e:
            log.warning("Could not reset LLM prefix cache: %s", e)
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
        if (
            payload is None
            or "hidden_states" not in payload
            or not payload["hidden_states"]
        ):
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
            return None

        full = [torch.cat([p, s], dim=0) for p, s in zip(prefix_hs, hs_suffix)]
        out = dict(payload)
        out["hidden_states"] = full
        return out

    def _gen_hidden_states(self, full_token_ids):
        """
        Path 1: Generate hidden states using VllmHiddenStatesGenerator.
        Try normal generation first.
        If got_len < expected_len (APC reused prefix), stitch missing prefix from local cache.
        If still unresolved, recompute unresolved sequences after resetting HS prefix cache.
        """
        hs_gen = self._get_hs_generator()
        try:
            raw_payload = hs_gen.generate(full_token_ids)
        except Exception as e:
            log.warning(e)
            raw_payload = None

        payload: List[Optional[dict]] = [None] * len(full_token_ids)
        unresolved: List[int] = []

        # Pass 1: stitch from local cache
        for i, req_ids in enumerate(full_token_ids):
            pay = (
                raw_payload[i]
                if raw_payload is not None and i < len(raw_payload)
                else None
            )
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
                len(unresolved),
                len(full_token_ids),
            )

            for i in unresolved:
                req_ids = full_token_ids[i]

                # Reset only HS path cache, not main LLM engine behavior
                self._reset_hs_prefix_cache(hs_gen)

                # Recompute single sequence (avoid batch-level cross-contamination)
                n_tries = 5
                one = None
                while n_tries > 0:
                    try:
                        n_tries -= 1
                        one = hs_gen.generate([req_ids])
                        break
                    except Exception as e:
                        log.error(e)
                    if n_tries <= 3:
                        log.warning("Reloading HS generator")
                        self._hs_gen = None
                        hs_gen = self._get_hs_generator()

                if one is None:
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

        # ---- Setup HS capture BEFORE generation if using native path ----
        if self.output_hidden_states and self.use_native_hs_capture:
            self._ensure_hs_extension()
            self._engine_core.collective_rpc("_reset_capture")

        # ---- 1) MAIN GENERATION (with HS capture if native path) ----
        outputs = self.llm.generate(prompts_list, sampling_params)

        # ---- 2) Extract hidden states ----
        hs_by_req_out: List[List[Optional[List[torch.Tensor]]]] = [
            [None] * len(ro.outputs) for ro in outputs
        ]
        context_lengths = None

        if self.output_hidden_states:
            prompt_token_ids = self._prompts_to_token_ids(prompts_list)
            context_lengths = [len(ids) for ids in prompt_token_ids]

            flat_full_ids, flat_index, _ = self._build_flat_full_token_ids(
                prompts_list, outputs
            )

            if self.use_native_hs_capture:
                # Path 2: Native capture — HS were captured during main generation.
                # Extract per-request hidden states and fill prefix-cache gaps.

                per_rank = self._engine_core.collective_rpc("_get_captured_states")
                captured_raw: dict = per_rank[0] if per_rank else {}
                meta_raw = self._engine_core.collective_rpc("_get_capture_metadata")
                meta: dict = meta_raw[0] if meta_raw else {}

                # Map output request_id to prompt index in prompts_list.
                # vLLM request ID formats:
                #   "6"         — simple numeric
                #   "6-abc123"  — numeric prefix + hash suffix
                #   "2_0"       — format {seq_idx}_{prompt_idx} for best-of-n
                output_short_ids: Dict[str, int] = {}
                for i, out in enumerate(outputs):
                    output_short_ids[out.request_id] = i
                    # Index by numeric prefix: "6-ab" -> "6"
                    if "-" in out.request_id:
                        prefix = out.request_id.split("-")[0]
                        output_short_ids.setdefault(prefix, i)
                    # Index by suffix for {seq}_{prompt} format: "0_1" -> "1"
                    if "_" in out.request_id:
                        suffix = out.request_id.rsplit("_", 1)[-1]
                        output_short_ids.setdefault(suffix, i)

                def _resolve_prompt_idx(req_id: str) -> Optional[int]:
                    # Exact match
                    if req_id in output_short_ids:
                        return output_short_ids[req_id]
                    # Format {seq}_{prompt}: suffix is prompt index
                    if "_" in req_id:
                        suffix = req_id.rsplit("_", 1)[-1]
                        if suffix in output_short_ids:
                            return output_short_ids[suffix]
                    # Format {prompt}-{hash}: prefix is prompt index
                    if "-" in req_id:
                        prefix = req_id.split("-")[0]
                        if prefix in output_short_ids:
                            return output_short_ids[prefix]
                    return None

                def _resolve_seq_idx(req_id: str) -> Optional[int]:
                    """Extract sequence index: '2_0' -> seq=2 (prefix)."""
                    if "_" in req_id:
                        parts = req_id.rsplit("_", 1)
                        try:
                            return int(parts[0])
                        except ValueError:
                            pass
                    return None

                # Process each captured layer
                for lid in self.hs_layer_ids:
                    layer_data = captured_raw.get(lid, {})
                    if not layer_data:
                        continue

                    # Deserialize per-request arrays
                    captured_arrays: Dict[str, np.ndarray] = {}
                    for req_id, pickled_bytes in layer_data.items():
                        captured_arrays[req_id] = pickle.loads(pickled_bytes)

                    # Build prompt groups and prompt_tokens for gap filling
                    prompt_groups: Dict[str, List[str]] = {}
                    req_prompt_tokens: Dict[str, List[int]] = {}
                    for req_id in captured_arrays:
                        idx = _resolve_prompt_idx(req_id)
                        if idx is None:
                            continue
                        text = prompts_list[idx]
                        prompt_groups.setdefault(text, []).append(req_id)
                        req_prompt_tokens[req_id] = prompt_token_ids[idx]

                    # Fill prefix-cache gaps
                    filled = _fill_prefix_gaps(
                        captured_arrays,
                        meta,
                        prompt_groups,
                        prompt_tokens=req_prompt_tokens,
                    )

                    # Assign filled HS to the right (request, output) slot.
                    # With best-of-n, vLLM creates sub-requests like "4_0",
                    # "4_1" — each with its own hidden states.  Map the suffix
                    # to the correct output sequence index j.
                    for req_id, arr in filled.items():
                        idx = _resolve_prompt_idx(req_id)
                        if idx is None:
                            continue

                        seq_j = _resolve_seq_idx(req_id)
                        num_outputs = len(outputs[idx].outputs)

                        if seq_j is not None and seq_j < num_outputs:
                            target_js = [seq_j]
                        else:
                            target_js = list(range(num_outputs))

                        for j in target_js:
                            out = outputs[idx].outputs[j]
                            gen_len = len(getattr(out, "token_ids", []) or [])
                            expected = len(prompt_token_ids[idx]) + gen_len

                            # Multi-step: if arr is short (prefix was cached
                            # from a previous step), prepend from step cache.
                            if arr.shape[0] < expected:
                                cached_layers = self._find_cached_hs(prompts_list[idx])
                                if cached_layers is not None:
                                    if lid in cached_layers:
                                        gap = expected - arr.shape[0]
                                        stored = cached_layers[lid]
                                        fillable = min(gap, stored.shape[0])
                                        if fillable > 0:
                                            arr = np.concatenate(
                                                [stored[:fillable], arr],
                                                axis=0,
                                            )

                            tensor = torch.from_numpy(arr)

                            # Trim if captured more than expected
                            if tensor.shape[0] > expected:
                                tensor = tensor[:expected]

                            # Pad with zero vectors if short (±1 token)
                            if tensor.shape[0] < expected:
                                pad_count = expected - tensor.shape[0]
                                pad = torch.zeros(
                                    pad_count, tensor.shape[1], dtype=tensor.dtype
                                )
                                tensor = torch.cat([tensor, pad], dim=0)

                            if hs_by_req_out[idx][j] is None:
                                hs_by_req_out[idx][j] = {"hidden_states": []}
                            hs_by_req_out[idx][j]["hidden_states"].append(tensor)

                            # Update step cache: use the current prompt
                            # TEXT as the key.  The next step's prompt is
                            # always an extension of the current prompt
                            # (prompt + trajectory), so string prefix
                            # matching is guaranteed to work — no BPE or
                            # stop-token issues.
                            actual = arr[: len(prompt_token_ids[idx]) + gen_len]
                            self._update_hs_step_cache(prompts_list[idx], lid, actual)

                # Cleanup stale cache entries.
                active_texts: List[str] = []
                for i_out, ro in enumerate(outputs):
                    for co in ro.outputs:
                        co_text = getattr(co, "text", "") or ""
                        active_texts.append(prompts_list[i_out] + co_text)
                self._cleanup_hs_step_cache(active_texts)

                # Cleanup hooks
                self._engine_core.collective_rpc(
                    "_setup_hidden_states_capture", args=([],)
                )
                self._engine_core.collective_rpc("_reset_capture")
                self._hs_extension_ready = False
                log.info("Native HS capture: extracted per-request hidden states")
            else:
                # Path 1: VllmHiddenStatesGenerator - separate generation
                # Sleep main LLM to free GPU memory before HS generator loads
                log.info(
                    "Sleeping main LLM (level=2) to free GPU memory before HS generator"
                )
                self.llm.sleep(level=2)

                if flat_full_ids:
                    flat_payloads = self._gen_hidden_states(flat_full_ids)
                    for k, payload in enumerate(flat_payloads):
                        i, j = flat_index[k]
                        # Convert dict payload to list of tensors format
                        if payload is not None and "hidden_states" in payload:
                            hs_by_req_out[i][
                                j
                            ] = payload  # Keep the whole dict with "hidden_states" key
                        else:
                            hs_by_req_out[i][j] = None

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

        for i, request_output in tqdm(
            enumerate(outputs), total=len(outputs), desc="Computing uncertainty"
        ):
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
                    out.prompt_token_ids = request_output.prompt_token_ids

                    # prompt_logprobs can be on request_output depending on vLLM version
                    prompt_lps = getattr(out, "prompt_logprobs", None)
                    if prompt_lps is None:
                        prompt_lps = getattr(request_output, "prompt_logprobs", None)

                    deps = {
                        "vllm_output": out,
                        "token_ids": getattr(out, "token_ids", None),
                        "logprobs": getattr(out, "logprobs", None),
                        "prompt_logprobs": prompt_lps,
                        "context_lengths": (
                            [context_lengths[i]]
                            if context_lengths is not None
                            else None
                        ),
                    }

                    if out_hs is not None:
                        deps["vllm_hidden_states_output"] = out_hs

                    for calc in self.stat_calculators:
                        deps.update(
                            calc(
                                deps,
                                texts=[prompts_list[i]],
                                model=self,
                                max_new_tokens=getattr(sampling_params, "max_tokens", 0)
                                or 0,
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
            prompts,
            sampling_params,
        )
        results = self._construct_request_with_uncertainty(
            prompts_list,
            sampling_params,
            outputs,
            hs_by_req_out,
            context_lengths,
            compute_uncertainty,
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
            deps["claims"] = [
                [
                    Claim(
                        None,
                        None,
                        list(range(claim_range[0], claim_range[1])),
                    )
                ]
            ]
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
