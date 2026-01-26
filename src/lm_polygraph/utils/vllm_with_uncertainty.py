from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import numpy as np
from lm_polygraph.estimators import Estimator
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# Optional dependency (only needed if output_hidden_states=True)
try:
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator
except Exception:
    VllmHiddenStatesGenerator = None


@dataclass
class RequestOutputWithUncertainty:
    """Extends vLLM RequestOutput to include uncertainty scores + optional hidden states payload."""
    request_output: RequestOutput
    uncertainty_scores: List[float]  # One score per output sequence
    hidden_states_payload: Optional[List[Dict[str, Any]]] = None
    # hidden_states_payload: list aligned with prompts:
    #   each item looks like {"input_ids": [...], "hidden_states": [Tensor,...], "loss_mask": [...]}

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
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.stat_calculators = stat_calculators
        self.estimator = estimator
        self.n_logprobs = n_logprobs

        self.output_hidden_states = output_hidden_states
        self.hs_layer_ids = hs_layer_ids or []
        self.hs_generator_kwargs = hs_generator_kwargs or {}

        self._hs_gen = None  # lazy init

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

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        compute_uncertainty: bool = True,
    ) -> List[RequestOutputWithUncertainty]:

        prompts_list = self._normalize_prompts(prompts)

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Enforce enough logprobs for calculators that need top-N
        sampling_params.logprobs = max(sampling_params.logprobs or 0, self.n_logprobs)
        if sampling_params.logprobs == 0:
            sampling_params.logprobs = 20  # fallback (your previous behavior)

        # ---- 1) Generate with vLLM (logprobs / tokens) ----
        outputs = self.llm.generate(prompts_list, sampling_params)

        # ---- 2) Optionally compute hidden states for prompts (or prompt+completion) ----
        hidden_states_payload = None
        if self.output_hidden_states:
            batch_token_ids = self._prompts_to_token_ids(prompts_list)
            hs_gen = self._get_hs_generator()
            hidden_states_payload = hs_gen.generate(batch_token_ids)

        # ---- 3) Compute uncertainty and package ----
        results: List[RequestOutputWithUncertainty] = []
        for request_output in outputs:
            request_scores = []
            prompt_hs = None
            if hidden_states_payload is not None and i < len(hidden_states_payload):
                prompt_hs = hidden_states_payload[i]   # dict for this prompt

            if compute_uncertainty:
                for out in request_output.outputs:
                    deps = {"vllm_output": out, "token_ids": out.token_ids, "logprobs": out.logprobs}
                    # pass hidden states into deps for calculators
                    if prompt_hs is not None:
                        deps["vllm_hidden_states_output"] = prompt_hs

                    # Run calculators + estimator
                    for calc in self.stat_calculators:
                        deps.update(calc(deps, texts=[prompts_list[i]], model=self,
                                         max_new_tokens=sampling_params.max_tokens if hasattr(sampling_params,
                                                                                              'max_tokens') else 0))

                    u = self.estimator(deps)
                    request_scores.append(float(u[0]))

            results.append(
                RequestOutputWithUncertainty(
                    request_output=request_output,
                    uncertainty_scores=request_scores,
                    hidden_states_payload=[prompt_hs] if prompt_hs is not None else None,
                )
            )
        return results

    def _score_from_deps(self, deps: Dict, prompt_hidden_states: Optional[Dict[str, Any]]) -> float:
        # Merge hidden states into deps if provided
        if prompt_hidden_states is not None:
            deps = dict(deps)
            deps["vllm_hidden_states_output"] = prompt_hidden_states
            # your VLLMHiddenStates StatCalculator can consume this key

        for calc in self.stat_calculators:
            deps.update(calc(deps))

        uncertainty = self.estimator(deps)
        return float(uncertainty[0])

    def score(
        self,
        token_ids: List[int],
        logprobs: List[Dict],
        hidden_states_output: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score arbitrary (possibly truncated) sequences.
        If you have hidden states from the generator for the same tokens, pass them in.
        """
        if not token_ids or not logprobs:
            return 0.0

        deps = {"token_ids": token_ids, "logprobs": logprobs}
        return self._score_from_deps(deps, prompt_hidden_states=hidden_states_output)

    def get_tokenizer(self):
        return self.tokenizer

    def __getattr__(self, name):
        return getattr(self.llm, name)
