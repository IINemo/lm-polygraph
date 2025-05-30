import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


def get_embeddings_from_output(
    output,
    batch,
    model_type,
    hidden_state: List[str] = ["encoder", "decoder"],
    ignore_padding: bool = True,
    use_averaging: bool = True,
    all_layers: bool = False,
    aggregation_method: str = "mean",
    level: str = "sequence",
    hidden_layer: int = -1,
):
    batch_embeddings = None
    batch_embeddings_decoder = None
    batch_size = len(batch["input_ids"])

    if model_type in ["CausalLM", "VisualLM"]:
        input_tokens_hs = output.hidden_states[0][hidden_layer].cpu().detach()
        if not all_layers:
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [h[hidden_layer].cpu().detach() for h in output.hidden_states[1:]],
                    dim=1,
                )
        else:
            input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [
                        h[hidden_layer].mean(axis=0).cpu().detach()
                        for h in output.hidden_states[1:]
                    ],
                    dim=1,
                )
        if len(output.hidden_states) > 1:
            if level == "sequence":
                batch_embeddings_decoder = (
                    torch.cat([input_tokens_hs, generated_tokens_hs], dim=1)
                    .mean(axis=1)
                    .cpu()
                    .detach()
                )
            elif level == "token":
                batch_embeddings_decoder = (
                    torch.cat([input_tokens_hs[:, -1:], generated_tokens_hs], dim=1)
                    .cpu()
                    .detach()
                )
        else:
            batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()

        if hasattr(output, "vision_hidden_states"):
            vision_features = output.vision_hidden_states[-1].cpu().detach()
            vision_embeddings = vision_features.mean(dim=1)
            if batch_embeddings is not None:
                batch_embeddings = torch.cat(
                    [batch_embeddings, vision_embeddings], dim=-1
                )
            else:
                batch_embeddings = vision_embeddings

    elif model_type == "Seq2SeqLM":
        if use_averaging:
            if "decoder" in hidden_state:
                try:
                    decoder_hidden_states = torch.stack(
                        [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                    )
                    if all_layers:
                        agg_decoder_hidden_states = decoder_hidden_states[
                            :, :, :, 0
                        ].mean(axis=1)
                    else:
                        agg_decoder_hidden_states = decoder_hidden_states[:, -1, :, 0]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=0
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(batch_size, -1, agg_decoder_hidden_states.shape[-1])[
                            :, 0
                        ]
                    )
                except TypeError:
                    if all_layers:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        ).mean(axis=0)
                    else:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        )[-1]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=1
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(-1, agg_decoder_hidden_states.shape[-1])
                    )

            if "encoder" in hidden_state:
                mask = batch["attention_mask"][:, :, None].cpu().detach()
                seq_lens = batch["attention_mask"].sum(-1)[:, None].cpu().detach()
                if all_layers:
                    encoder_embeddings = (
                        aggregate(
                            torch.stack(output.encoder_hidden_states), "mean", axis=0
                        )
                        .cpu()
                        .detach()
                        * mask
                    )
                else:
                    encoder_embeddings = (
                        output.encoder_hidden_states[-1].cpu().detach() * mask
                    )

                if ignore_padding:
                    if aggregation_method == "mean":
                        batch_embeddings = (encoder_embeddings).sum(
                            1
                        ).cpu().detach() / seq_lens
                    else:
                        batch_embeddings = (
                            aggregate(encoder_embeddings, aggregation_method, axis=1)
                            .cpu()
                            .detach()
                        )
                else:
                    batch_embeddings = (
                        aggregate(encoder_embeddings, aggregation_method, axis=1)
                        .cpu()
                        .detach()
                    )
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
        else:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                last_decoder_hidden_states = decoder_hidden_states[-1, -1, :, 0]
                batch_embeddings_decoder = (
                    last_decoder_hidden_states.reshape(
                        batch_size, -1, last_decoder_hidden_states.shape[-1]
                    )[:, 0]
                    .cpu()
                    .detach()
                )
            if "encoder" in hidden_state:
                batch_embeddings = output.encoder_hidden_states[-1][:, 0].cpu().detach()
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
    else:
        raise NotImplementedError

    return batch_embeddings, batch_embeddings_decoder


def aggregate(x, aggregation_method, axis):
    if aggregation_method == "max":
        return x.max(axis=axis).values
    elif aggregation_method == "mean":
        return x.mean(axis=axis)
    elif aggregation_method == "sum":
        return x.sum(axis=axis)


class EmbeddingsCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["train_embeddings", "background_train_embeddings"], []

    def __init__(self, hidden_layer: int = -1):
        super().__init__()
        self.hidden_layer = hidden_layer

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=False,
                output_hidden_states=True,
                num_beams=1,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type
            )

        if model.model_type in ["CausalLM", "VisualLM"]:
            return {
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        elif model.model_type == "Seq2SeqLM":
            return {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        else:
            raise NotImplementedError
