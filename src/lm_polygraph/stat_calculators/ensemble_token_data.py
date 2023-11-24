from typing import Dict, List
from functools import partial

import torch
import numpy as np
from transformers import PreTrainedModel

from .stat_calculator import StatCalculator
from lm_polygraph.utils.token_restoration import (
    get_collect_fn,
)


class EnsembleTokenLevelDataCalculator(StatCalculator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: PreTrainedModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        ensemble_model = dependencies["ensemble_model"]

        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        batch = {k: v.to(ensemble_model.device()) for k, v in batch.items()}
        generation_params = dependencies["ensemble_generation_params"]

        max_length = generation_params.get("generation_max_length", max_new_tokens)
        min_length = generation_params.get("generation_min_length", 2)
        num_return_sequences = generation_params.get("num_return_sequences", 5)

        model_config = ensemble_model.model.config
        if "mbart" in model_config._name_or_path:
            model_config.decoder_start_token_id = model.tokenizer.lang_code_to_id[
                model.tokenizer.tgt_lang
            ]

        if generation_params.get("num_beams") is None and (
            "do_sample" not in generation_params
            or generation_params["do_sample"] is None
        ):
            generation_params["num_beams"] = num_return_sequences

        with torch.no_grad():
            output = ensemble_model.generate(
                **batch,
                max_length=max_length,
                min_length=min_length,
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=num_return_sequences,
                **generation_params,
            )

        batch_length = len(batch["input_ids"])

        collect_fn = get_collect_fn(output)
        collect_fn = partial(
            collect_fn,
            output,
            batch_length,
            num_return_sequences,
            ensemble_model.model.config.vocab_size,
            ensemble_model.model.config.pad_token_id,
        )

        pe_token_level_scores = collect_fn(
            ensemble_uncertainties=output["pe_uncertainties"]
        )
        ep_token_level_scores = collect_fn(
            ensemble_uncertainties=output["ep_uncertainties"]
        )

        output_dict = {
            "pe_token_level_scores": pe_token_level_scores,
            "ep_token_level_scores": ep_token_level_scores,
            "probas": pe_token_level_scores["probas"],
            "log_probas": pe_token_level_scores["log_probas"],
        }

        if ensemble_model.model.ensembling_mode == "pe":
            output_dict.update(
                {
                    "weights": torch.Tensor(pe_token_level_scores["weights"]),
                    "scores_unbiased": torch.Tensor(
                        pe_token_level_scores["scores_unbiased"]
                    ),
                    "entropy": torch.Tensor(pe_token_level_scores["entropy"]),
                    "entropy_top5": torch.Tensor(pe_token_level_scores["entropy_top5"]),
                    "entropy_top10": torch.Tensor(
                        pe_token_level_scores["entropy_top10"]
                    ),
                    "entropy_top15": torch.Tensor(
                        pe_token_level_scores["entropy_top15"]
                    ),
                }
            )
        elif ensemble_model.model.ensembling_mode == "ep":
            output_dict.update(
                {
                    "weights": torch.Tensor(ep_token_level_scores["weights"]),
                    "scores_unbiased": torch.Tensor(
                        ep_token_level_scores["scores_unbiased"]
                    ),
                    "entropy": torch.Tensor(ep_token_level_scores["entropy"]),
                    "entropy_top5": torch.Tensor(ep_token_level_scores["entropy_top5"]),
                    "entropy_top10": torch.Tensor(
                        ep_token_level_scores["entropy_top10"]
                    ),
                    "entropy_top15": torch.Tensor(
                        ep_token_level_scores["entropy_top15"]
                    ),
                }
            )
        else:
            raise NotImplementedError

        return {"ensemble_token_scores": output_dict}
