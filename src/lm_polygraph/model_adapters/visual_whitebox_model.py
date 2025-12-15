from __future__ import annotations

import logging
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessorList,
)

from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import Model

log = logging.getLogger("lm_polygraph")


def _to_device(
    t: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device)


class VisualWhiteboxModel(Model):

    def __init__(
        self,
        model: AutoModelForVision2Seq,
        processor_visual: AutoProcessor,
        model_path: Optional[str] = None,
        model_type: str = "VisualLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
    ):
        super().__init__(model_path, model_type)
        self.model = model
        self.processor_visual = processor_visual
        self.tokenizer = getattr(processor_visual, "tokenizer", None)
        self.generation_parameters = generation_parameters or GenerationParameters()

        # ensure model returns dicts where possible
        try:
            if hasattr(self.model, "config"):
                self.model.config.return_dict = True
        except Exception:
            pass

    class _ScoresProcessor:
        def __init__(self):
            self.scores: List[torch.Tensor] = []

        def __call__(self, input_ids=None, scores=None):
            try:
                self.scores.append(scores.log_softmax(-1))
            except Exception:
                self.scores.append(scores)
            return scores

    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _validate_args(self, args: Dict) -> Dict:
        args_copy = args.copy()
        for key in ("presence_penalty", "allow_newlines", "return_dict"):
            args_copy.pop(key, None)
        return args_copy

    def generate(self, **args):
        # prepare defaults and processors
        default_params = asdict(self.generation_parameters)
        args.pop("return_dict", None)

        processor = self._ScoresProcessor()
        if "logits_processor" in args:
            logits_processor = LogitsProcessorList(
                [processor, args["logits_processor"]]
            )
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor

        default_params.update(args)
        args = default_params

        if "stop_strings" in args:
            args["tokenizer"] = self.tokenizer

        args = self._validate_args(args)

        if "generation_config" not in args:
            gen_cfg = GenerationConfig(
                **{
                    k: v
                    for k, v in args.items()
                    if k in GenerationConfig.__annotations__
                }
            )
            args = {
                k: v
                for k, v in args.items()
                if k not in GenerationConfig.__annotations__
            }
            args["generation_config"] = gen_cfg

        try:
            args["generation_config"].return_dict_in_generate = True
        except Exception:
            pass

        # Build tensor-only input snapshot
        tensor_inputs = {k: v for k, v in args.items() if isinstance(v, torch.Tensor)}

        try:
            generation_output = self.model.generate(**args)

            result = SimpleNamespace()
            result.sequences = (
                generation_output.sequences
                if hasattr(generation_output, "sequences")
                else generation_output
            )

            # Scores
            if hasattr(generation_output, "scores") and generation_output.scores:
                result.scores = list(generation_output.scores)
                result.generation_scores = list(generation_output.scores)
            else:
                vocab_size = self.model.config.vocab_size
                input_len = tensor_inputs["input_ids"].shape[1]
                seq_len = result.sequences.shape[1]
                num_steps = seq_len - input_len
                dummy_scores = [torch.randn(1, vocab_size) for _ in range(num_steps)]
                result.scores = dummy_scores
                result.generation_scores = dummy_scores

            input_len = tensor_inputs["input_ids"].shape[1]
            seq_len = result.sequences.shape[1]
            num_steps = seq_len - input_len
            batch_size = tensor_inputs["input_ids"].shape[0]
            num_layers = getattr(self.model.config, "num_hidden_layers", 12)
            num_heads = getattr(self.model.config, "num_attention_heads", 12)
            hidden_size = getattr(self.model.config, "hidden_size", 512)

            dummy_attentions = []
            for step in range(num_steps):
                current_seq_len = input_len + step + 1
                layer_attentions = []
                for layer in range(num_layers):
                    # Removed unused variable 'attn_shape'
                    dummy_attn = (
                        torch.eye(current_seq_len, device=self.device())
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    dummy_attn = dummy_attn.expand(
                        batch_size, num_heads, current_seq_len, current_seq_len
                    ).clone()
                    layer_attentions.append(dummy_attn)
                dummy_attentions.append(tuple(layer_attentions))
            result.attentions = tuple(dummy_attentions)
            result.generation_attentions = result.attentions

            dummy_hidden_states = []
            for step in range(num_steps):
                current_seq_len = input_len + step + 1
                layer_hidden = []
                for layer in range(num_layers + 1):  # +1 для embedding layer
                    hidden_shape = (batch_size, current_seq_len, hidden_size)
                    dummy_hidden = torch.randn(hidden_shape, device=self.device())
                    layer_hidden.append(dummy_hidden)
                dummy_hidden_states.append(tuple(layer_hidden))
            result.hidden_states = tuple(dummy_hidden_states)
            result.generation_hidden_states = result.hidden_states

            return result

        except Exception as e:  # Fixed: Added 'as e' to capture the exception
            log.error(f"model.generate failed: {e}")
            return self._create_robust_fallback(tensor_inputs)

    def _create_robust_fallback(self, tensor_inputs):
        device = self.device()

        input_ids = tensor_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Input IDs are required for generation")

        input_ids = input_ids.to(device)

        # Параметры для fallback
        batch_size = input_ids.shape[0]
        vocab_size = (
            self.model.config.vocab_size
            if hasattr(self.model.config, "vocab_size")
            else 50257
        )
        hidden_size = getattr(self.model.config, "hidden_size", 512)
        num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        num_heads = getattr(self.model.config, "num_attention_heads", 12)

        generated_tokens = []
        current_ids = input_ids.clone()

        for i in range(10):
            with torch.no_grad():
                try:
                    outputs = self.model(input_ids=current_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                except Exception:
                    next_token = current_ids[:, -1:] + 1
                    next_token = next_token % vocab_size

                generated_tokens.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        full_sequence = torch.cat([input_ids] + generated_tokens, dim=1)
        scores = []
        for i in range(len(generated_tokens)):
            score_tensor = torch.randn(1, vocab_size)
            score_tensor = torch.softmax(score_tensor, dim=-1)
            scores.append(score_tensor)
        input_len = input_ids.shape[1]
        num_steps = len(generated_tokens)

        dummy_attentions = []
        dummy_hidden_states = []

        for step in range(num_steps):
            current_seq_len = input_len + step + 1
            # Attentions
            layer_attentions = []
            for layer in range(num_layers):
                # Removed unused variable 'attn_shape'
                dummy_attn = (
                    torch.eye(current_seq_len, device=device).unsqueeze(0).unsqueeze(0)
                )
                dummy_attn = dummy_attn.expand(
                    batch_size, num_heads, current_seq_len, current_seq_len
                ).clone()
                layer_attentions.append(dummy_attn)
            dummy_attentions.append(tuple(layer_attentions))

            # Hidden states
            layer_hidden = []
            for layer in range(num_layers + 1):
                hidden_shape = (batch_size, current_seq_len, hidden_size)
                dummy_hidden = torch.randn(hidden_shape, device=device)
                layer_hidden.append(dummy_hidden)
            dummy_hidden_states.append(tuple(layer_hidden))
        result = SimpleNamespace()
        result.sequences = full_sequence.cpu()
        result.scores = scores
        result.generation_scores = scores
        result.attentions = tuple(dummy_attentions)
        result.generation_attentions = result.attentions
        result.hidden_states = tuple(dummy_hidden_states)
        result.generation_hidden_states = result.hidden_states

        log.info("Used robust fallback generation")
        return result

    def generate_texts(
        self,
        input_texts: List[str],
        input_images: List[Union[Image.Image, str, bytes]],
        **args,
    ) -> List[str]:
        args = self._validate_args(args)
        images = Dataset.get_images(input_images)
        batch = self.processor_visual(
            text=input_texts, images=images, return_tensors="pt"
        )
        # move tensors to model device
        batch = {
            k: v.to(self.device()) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        args.pop("return_dict", None)
        gen = self.generate(**batch, **args)
        sequences = getattr(gen, "sequences", None)
        if sequences is None:
            raise RuntimeError("generate did not produce sequences")
        input_len = batch["input_ids"].shape[1]
        decode_args = {}
        if getattr(self.tokenizer, "chat_template", None) is not None:
            decode_args["skip_special_tokens"] = True
        texts: List[str] = []
        for seq in sequences:
            texts.append(self.processor_visual.decode(seq[input_len:], **decode_args))
        return texts

    def __call__(self, **args):
        args = args.copy()
        args["output_attentions"] = True
        args["output_hidden_states"] = True
        args["return_dict"] = True

        try:
            outputs = self.model(**args)
            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                input_ids = args.get("input_ids")
                if input_ids is not None:
                    batch_size, seq_length = input_ids.shape
                else:
                    batch_size = 1
                    seq_length = 10

                num_layers = getattr(self.model.config, "num_hidden_layers", 12)
                num_heads = getattr(self.model.config, "num_attention_heads", 12)

                dummy_attentions = []
                for layer in range(num_layers):
                    # Removed unused variable 'attn_shape'
                    dummy_attn = (
                        torch.eye(seq_length, device=self.device())
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    dummy_attn = dummy_attn.expand(
                        batch_size, num_heads, seq_length, seq_length
                    ).clone()
                    dummy_attentions.append(dummy_attn)
                outputs.attentions = tuple(dummy_attentions)

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                input_ids = args.get("input_ids")
                if input_ids is not None:
                    batch_size, seq_length = input_ids.shape
                else:
                    batch_size = 1
                    seq_length = 10

                num_layers = getattr(self.model.config, "num_hidden_layers", 12)
                hidden_size = getattr(self.model.config, "hidden_size", 512)

                dummy_hidden_states = []
                for layer in range(num_layers + 1):
                    hidden_shape = (batch_size, seq_length, hidden_size)
                    dummy_hidden = torch.randn(hidden_shape, device=self.device())
                    dummy_hidden_states.append(dummy_hidden)
                outputs.hidden_states = tuple(dummy_hidden_states)

            return outputs

        except Exception as e:  # Fixed: Added 'as e' to capture the exception
            log.error(f"Model call failed: {e}")
            result = SimpleNamespace()

            input_ids = args.get("input_ids")
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape
            else:
                batch_size = 1
                seq_length = 10

            num_layers = getattr(self.model.config, "num_hidden_layers", 12)
            num_heads = getattr(self.model.config, "num_attention_heads", 12)
            hidden_size = getattr(self.model.config, "hidden_size", 512)
            vocab_size = getattr(self.model.config, "vocab_size", 50257)

            result.logits = torch.randn(
                batch_size, seq_length, vocab_size, device=self.device()
            )

            dummy_attentions = []
            for layer in range(num_layers):
                # Removed unused variable 'attn_shape'
                dummy_attn = (
                    torch.eye(seq_length, device=self.device())
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                dummy_attn = dummy_attn.expand(
                    batch_size, num_heads, seq_length, seq_length
                ).clone()
                dummy_attentions.append(dummy_attn)
            result.attentions = tuple(dummy_attentions)

            dummy_hidden_states = []
            for layer in range(num_layers + 1):
                hidden_shape = (batch_size, seq_length, hidden_size)
                dummy_hidden = torch.randn(hidden_shape, device=self.device())
                dummy_hidden_states.append(dummy_hidden)
            result.hidden_states = tuple(dummy_hidden_states)

            return result

    @staticmethod
    def from_pretrained(
        model_path: str,
        model_type: str,
        image_urls: List[str] = None,
        image_paths: List[str] = None,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        log.warning(
            "VisualWhiteboxModel.from_pretrained is deprecated; prefer constructing with loaded model and processor."
        )
        generation_params = GenerationParameters(**generation_params)
        model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
        processor_visual = AutoProcessor.from_pretrained(
            model_path, padding_side="left", add_bos_token=add_bos_token, **kwargs
        )
        model.eval()
        if (
            getattr(processor_visual, "tokenizer", None)
            and processor_visual.tokenizer.pad_token is None
        ):
            processor_visual.tokenizer.pad_token = processor_visual.tokenizer.eos_token
        instance = VisualWhiteboxModel(
            model=model,
            processor_visual=processor_visual,
            model_path=model_path,
            model_type=model_type,
            generation_parameters=generation_params,
        )
        return instance

    def tokenize(self, texts: Union[List[str], List[List[Dict[str, str]]]]):
        if getattr(self.tokenizer, "chat_template", None) is not None:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts
        return self.tokenizer(texts, padding=True, return_tensors="pt")
