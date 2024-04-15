import inspect
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import GenerationMixin
from transformers.generation.utils import ModelOutput

from lm_polygraph.utils.ensemble_utils.ensemble_beam import EnsembleBeamSearchMixin
from lm_polygraph.utils.ensemble_utils.ensemble_sample import EnsembleSampleMixin
from lm_polygraph.utils.ensemble_utils.ensemble_greedy import EnsembleGreedyMixin


class EnsembleGenerationMixin(
    EnsembleBeamSearchMixin, EnsembleSampleMixin, EnsembleGreedyMixin, GenerationMixin
):
    def add_ensemble_models(self, models, devices):
        self._models_list = list(models)

    @property
    def tokenizer(self):
        if hasattr(self, "_tokenizer"):
            return self._tokenizer
        return None

    @tokenizer.setter
    def tokenizer(self, value=None):
        self._tokenizer = value

    @property
    def models(self):
        return [self] + self._models_list

    @property
    def ensembling_mode(self):
        if self._ensembling_mode is not None:
            return self._ensembling_mode
        return "pe"

    @ensembling_mode.setter
    def ensembling_mode(self, value="pe"):
        self._ensembling_mode = value

    @property
    def mc(self):
        if hasattr(self, "_mc") and self._mc is not None:
            return self._mc
        return False

    @mc.setter
    def mc(self, value=False):
        self._mc = value

    @property
    def mc_models_num(self):
        if hasattr(self, "_mc_models_num"):
            return self._mc_models_num
        return 1

    @mc_models_num.setter
    def mc_models_num(self, num=1):
        self._mc_models_num = num

    @property
    def base_seed(self):
        return self._base_seed

    @base_seed.setter
    def base_seed(self, seed=42):
        self._base_seed = seed

    @property
    def mc_seeds(self):
        return self._mc_seeds

    @mc_seeds.setter
    def mc_seeds(self, seeds=[]):
        self._mc_seeds = seeds

    @property
    def models_beam_logits_iter(self):
        if hasattr(self, "_models_beam_logits_iter"):
            return self._models_beam_logits_iter

    @models_beam_logits_iter.setter
    def models_beam_logits_iter(self, value):
        self._models_beam_logits_iter = value

    def calculate_entropy_based_measures(self, enable=True):
        self.calculate_entropies = enable

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if getattr(self, "models", None) is None:
            self._models_list = []

        if self.mc:
            # 1. get encoders
            encoder = self.get_encoder()
            # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
            # as the inputs.
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True

            # 2. prepare encoder args and encoder kwargs from model kwargs
            irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not any(argument.startswith(p) for p in irrelevant_prefix)
            }

            encoder_signature = set(inspect.signature(encoder.forward).parameters)
            encoder_accepts_wildcard = (
                "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
            )
            if not encoder_accepts_wildcard:
                encoder_kwargs = {
                    argument: value
                    for argument, value in encoder_kwargs.items()
                    if argument in encoder_signature
                }

            # 3. make sure that encoder returns `ModelOutput`
            model_input_name = (
                model_input_name
                if model_input_name is not None
                else self.main_input_name
            )
            encoder_kwargs["return_dict"] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            outs = []
            for i in range(self.mc_models_num):
                torch.manual_seed(self.mc_seeds[i])
                outs.append(encoder(**encoder_kwargs))
            torch.manual_seed(self.base_seed)
            model_kwargs["encoder_outputs"]: List[ModelOutput] = outs
        else:
            model_kwargs["encoder_outputs"]: List[ModelOutput] = []
            for model in self.models:
                # 1. get encoder
                encoder = self.get_encoder()
                # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
                # as the inputs.
                if hasattr(encoder, "_hf_hook"):
                    encoder._hf_hook.io_same_device = True
                # 2. prepare encoder args and encoder kwargs from model kwargs
                irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
                encoder_kwargs = {
                    argument: value
                    for argument, value in model_kwargs.items()
                    if not any(argument.startswith(p) for p in irrelevant_prefix)
                }
                encoder_signature = set(inspect.signature(encoder.forward).parameters)
                encoder_accepts_wildcard = (
                    "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
                )
                if not encoder_accepts_wildcard:
                    encoder_kwargs = {
                        argument: value
                        for argument, value in encoder_kwargs.items()
                        if argument in encoder_signature
                    }

                # 3. make sure that encoder returns `ModelOutput`
                model_input_name = (
                    model_input_name
                    if model_input_name is not None
                    else self.main_input_name
                )
                encoder_kwargs["return_dict"] = True
                encoder_kwargs[model_input_name] = inputs_tensor

                encoder_kwargs[model_input_name].to(model.device)
                encoder_kwargs = {
                    k: v.to(model.device)
                    for k, v in encoder_kwargs.items()
                    if hasattr(v, "to")
                }
                model_kwargs["encoder_outputs"].append(encoder(**encoder_kwargs))

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(
                    dict_to_expand[key], torch.Tensor
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(
                        expand_size, dim=0
                    )
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs_expanded = []
            for output in model_kwargs["encoder_outputs"]:
                encoder_outputs_expanded.append(_expand_dict_for_generation(output))
            model_kwargs["encoder_outputs"] = encoder_outputs_expanded

        return input_ids, model_kwargs
