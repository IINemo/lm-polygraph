from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters
from transformers.generation import GenerateDecoderOnlyOutput

import torch
from typing import List


class WhiteboxModelvLLM(Model):
    """Basic whitebox model adapter for using vLLM in stat calculators and uncertainty estimators."""

    def __init__(
        self,
        model,
        sampling_params,
        generation_parameters: GenerationParameters = GenerationParameters(),
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling_params = sampling_params
        self.generation_parameters = generation_parameters

        self.sampling_params.stop = list(
            getattr(self.generation_parameters, "generate_until", list())
        )

        for param in [
            "presence_penalty",
            "repetition_penalty",
            "temperature",
            "top_p",
            "top_k",
        ]:
            setattr(
                self.sampling_params,
                param,
                getattr(self.generation_parameters, param, None),
            )

        self.base_device = device
        self.model_type = "vLLMCausalLM"

    def generate(self, *args, **kwargs):
        sampling_params = self.sampling_params
        sampling_params.n = kwargs.get("num_return_sequences", 1)
        texts = self.tokenizer.batch_decode(
            kwargs["input_ids"], skip_special_tokens=True
        )
        sampling_params.stop = []
        output = self.model.generate(*args, texts, sampling_params)
        return self.post_processing(output)

    def device(self):
        return self.base_device

    def tokenize(self, texts):
        output = self.tokenizer(texts, return_tensors="pt", padding=True)
        return output

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate_texts(self, input_texts: List[str], **args):
        outputs = self.generate(input_texts, **args)
        texts = [
            outputs.text
            for sampled_outputs in outputs
            for outputs in sampled_outputs.outputs
        ]
        return texts

    def post_processing(self, outputs):
        vocab_size = max(
            self.tokenizer.vocab_size, max(self.tokenizer.added_tokens_decoder.keys())
        )
        logits = []
        sequences = []

        max_seq_len = max(
            [
                len(output.token_ids)
                for sampled_outputs in outputs
                for output in sampled_outputs.outputs
            ]
        )
        for sample_output in outputs:

            for output in sample_output.outputs:

                log_prob = torch.zeros((max_seq_len, vocab_size)).fill_(-torch.inf)
                sequence = (
                    torch.zeros(max_seq_len).fill_(self.tokenizer.eos_token_id).long()
                )

                for i, probs in enumerate(output.logprobs):
                    top_tokens = torch.tensor(list(probs.keys()))
                    top_values = torch.tensor([lp.logprob for lp in probs.values()])
                    log_prob[i, top_tokens] = top_values
                    sequence[i] = output.token_ids[i]

                logits.append(log_prob)
                sequences.append(sequence)

        standard_output = GenerateDecoderOnlyOutput(
            sequences=sequences, logits=logits, scores=logits
        )
        return standard_output
