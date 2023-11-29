import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class BlackboxSamplingGenerationCalculator(StatCalculator):
    """
    Calculates several sampled texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self, samples_n: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        super().__init__(["blackbox_sample_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates sampled texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[str]] sampled texts at 'blackbox_sample_texts' key.
        """

        if isinstance(model, BlackboxModel):
            samples = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                temperature=model.parameters.temperature,
                top_p=model.parameters.topp,
                presence_penalty=model.parameters.presence_penalty,
                repetition_penalty=model.parameters.repetition_penalty,
                top_k=model.parameters.topk if model.parameters.topk > 1 else 50,
                n=self.samples_n,
            )
        else:
            samples = [[] for _ in range(len(texts))]
            out = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                min_length=2,
                do_sample=True,
                num_beams=1,
                temperature=model.parameters.temperature,
                top_p=model.parameters.topp,
                repetition_penalty=model.parameters.repetition_penalty,
                top_k=model.parameters.topk if model.parameters.topk > 1 else 50,
                num_return_sequences=self.samples_n,
            )
            for i in range(len(texts)):
                for j in range(self.samples_n):
                    samples[i].append(out[i * self.samples_n + j])

        return {
            "blackbox_sample_texts": samples,
        }


def _gen_samples(n_samples, model, batch, **kwargs):
    batch_size = len(batch["input_ids"])
    logits, sequences = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    with torch.no_grad():
        for k in range(n_samples):
            out = model.generate(**batch, **kwargs)
            cur_logits = torch.stack(out.scores, dim=1).log_softmax(-1)
            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    return sequences, sum(logits, [])


class SamplingGenerationCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * sampled texts
    * tokens of the sampled texts
    * probabilities of the sampled tokens generation
    """

    def __init__(self, samples_n: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        super().__init__(["sample_log_probs", "sample_tokens", "sample_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[List[float]]]): probabilities at each token of the sampling generation.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        sequences, logits = _gen_samples(
            self.samples_n,
            model,
            batch,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            min_length=2,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
        )

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        if model.model_type == "Seq2SeqLM":
            sequences = [seq[1:] for seq in sequences]
        for i in range(len(logits)):
            log_prob, toks = 0, []
            inp_size = (
                len(batch["input_ids"][int(i / self.samples_n)])
                if model.model_type == "CausalLM"
                else 0
            )
            for j in range(len(sequences[i]) - inp_size):
                cur_token = sequences[i][j + inp_size].item()
                log_prob += logits[i][j][cur_token].item()
                if cur_token == model.tokenizer.eos_token_id:
                    break
                toks.append(cur_token)
            log_probs[int(i / self.samples_n)].append(log_prob)
            tokens[int(i / self.samples_n)].append(toks)
            texts[int(i / self.samples_n)].append(model.tokenizer.decode(toks))
        return {
            "sample_log_probs": log_probs,
            "sample_tokens": tokens,
            "sample_texts": texts,
        }
