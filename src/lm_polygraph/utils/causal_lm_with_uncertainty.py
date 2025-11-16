from lm_polygraph.model_adapters import WhiteboxModelBasic
from transformers.generation.utils import GenerateDecoderOnlyOutput
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Dict, Any
import torch


@dataclass
class GenerateDecoderOnlyOutputWithUncertainty(GenerateDecoderOnlyOutput):
    """Extends GenerateDecoderOnlyOutput to include uncertainty scores"""

    uncertainty_score: Optional[Union[float, List[float], torch.Tensor]] = None
    deps: Optional[Dict[str, Any]] = None


class CausalLMWithUncertainty:
    def __init__(self, llm, tokenizer, stat_calculators, estimator, keep_deps=None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.stat_calculators = stat_calculators
        self.estimator = estimator
        self.keep_deps = keep_deps

    def generate(self, input_ids, attention_mask=None, **kwargs):
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.model_adapter = WhiteboxModelBasic(
            model=self.llm,
            tokenizer=self.tokenizer,
            tokenizer_args={
                "add_special_tokens": False,
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
            },
            model_type="CausalLM",
            generation_parameters=kwargs,
        )

        deps = dict()
        deps["model_inputs"] = {
            "input_ids": input_ids,
            **kwargs,
        }
        texts = self.tokenizer.batch_decode(input_ids)
        for calc in self.stat_calculators:
            deps.update(
                calc(
                    deps,
                    texts=texts,
                    model=self.model_adapter,
                    max_new_tokens=max_new_tokens,
                )
            )

        uncertainty_score = self.estimator(deps)

        raw_out = deps["out"]
        out_with_uncertainty = GenerateDecoderOnlyOutputWithUncertainty(
            **asdict(raw_out),
            uncertainty_score=uncertainty_score,
            deps={k: v for k, v in deps.items() if k in self.keep_deps} if self.keep_deps is not None else None
        )
        return out_with_uncertainty

    def device(self):
        return self.llm.device
