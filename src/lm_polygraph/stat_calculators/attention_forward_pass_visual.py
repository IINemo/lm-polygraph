import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class AttentionForwardPassCalculatorVisual(StatCalculator):
    """
    For VisualLM (vision-language) WhiteboxModel: computes attention weights
    during forward pass using greedy tokens + image inputs.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["forwardpass_attention_weights"], ["greedy_tokens"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
            dependencies: includes greedy_tokens.
            texts: List of dicts with keys like 'text'
            model: WhiteboxModel wrapper.
        Returns:
            Dict with 'forwardpass_attention_weights' numpy array.
        """
        # Tokenize image+text pairs
        batches = {}
        images = dependencies["images"]

        for text, image in zip(texts, images):
            batch = model.processor_visual(
                text=str(text),
                images=image,
                return_tensors="pt",
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
        cut_sequences = dependencies["greedy_tokens"]
        forwardpass_attention_weights = []

        for i in range(len(texts)):
            input_ids = batch["input_ids"][i].unsqueeze(0)
            vision_inputs = {
                k: v[i].unsqueeze(0) for k, v in batch.items() if k != "input_ids"
            }

            greedy_ids = torch.tensor([cut_sequences[i]], device=model.device())

            full_input_ids = torch.cat([input_ids, greedy_ids], dim=1)

            model_inputs = {"input_ids": input_ids, "output_attentions": True}

            if "pixel_values" in batch:
                model_inputs["pixel_values"] = batch["pixel_values"][i].unsqueeze(0)
            if "img_input_mask" in batch:
                model_inputs["img_input_mask"] = batch["img_input_mask"][i].unsqueeze(0)

            # with torch.no_grad():
            #     outputs = model.model(
            #         input_ids=input_ids,
            #         pixel_values=pixel_values,
            #         output_attentions=True,
            #         output_hidden_states=True
            #     )
            print(
                {
                    k: v.shape if isinstance(v, torch.Tensor) else type(v)
                    for k, v in model_inputs.items()
                }
            )

            for i in range(len(texts)):
                encoding = model.processor_visual(
                    text=str(texts[i]), images=images[i], return_tensors="pt"
                ).to(model.device())

                with torch.no_grad():
                    out = model.model(**encoding, output_attentions=True)
                    attentions = out.attentions
                    attentions = tuple(a.to("cpu") for a in attentions)
                    attentions_np = torch.cat(attentions).float().numpy()

                forwardpass_attention_weights.append(attentions_np)

            # with torch.no_grad():
            #     out = model.model(**model_inputs)
            #     attentions = out.attentions
            #     attentions = tuple(a.to("cpu") for a in attentions)
            #     attentions_np = torch.cat(attentions).float().numpy()

            # forwardpass_attention_weights.append(attentions_np)

        # Pad if sequence lengths mismatch
        try:
            forwardpass_attention_weights = np.array(forwardpass_attention_weights)
        except Exception:
            max_seq_length = max(el.shape[-1] for el in forwardpass_attention_weights)
            padded = []
            for el in forwardpass_attention_weights:
                if el.shape[-1] != max_seq_length:
                    pad_mask = (
                        (0, 0),
                        (0, 0),
                        (0, max_seq_length - el.shape[-1]),
                        (0, max_seq_length - el.shape[-1]),
                    )
                    el = np.pad(el, pad_mask, constant_values=np.nan)
                padded.append(el)
            forwardpass_attention_weights = np.array(padded)

        return {"forwardpass_attention_weights": forwardpass_attention_weights}
