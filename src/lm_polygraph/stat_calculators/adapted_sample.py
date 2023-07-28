import torch
import numpy as np
import math

from typing import Dict, List

from .stat_calculator import StatCalculator
from transformers import LogitsProcessor, LogitsProcessorList
from lm_polygraph.utils.model import WhiteboxModel


class EraseHypothesesLogitsProcessor(LogitsProcessor):
    def __init__(
        self, hyps_to_erase: List[List[int]], hyps_logprobs: List[float], input_len: int
    ):
        self.hyps_to_erase: List[List[int]] = hyps_to_erase
        self.hyps_logprobs: List[float] = hyps_logprobs
        self.importance_logits, self.logits = [], []
        self.input_len = input_len

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.LongTensor:
        self.logits.append(scores.clone())
        scores = scores.log_softmax(-1)
        for i in range(len(input_ids)):
            equal = True
            for j in range(len(input_ids[i]) - self.input_len):
                if (
                    len(self.hyps_to_erase[i]) < j + 1
                    or input_ids[i, j + self.input_len].item()
                    != self.hyps_to_erase[i][j]
                ):
                    equal = False
                    break
            if len(input_ids[i]) - self.input_len >= len(self.hyps_to_erase[i]):
                equal = False
            if equal:
                next_token = self.hyps_to_erase[i][len(input_ids[i]) - self.input_len]
                a, b = scores[i][next_token].item(), self.hyps_logprobs[i]
                # log(e^a - e^b) = b + (e^(a - b) - 1).log()
                if a < b - 1e-5:
                    scores[i][next_token] = -10000
                else:
                    if a - b >= 100:  # b + log(e^{a-b} - 1) ~ b + log(e^{a-b}) = b + a - b = a
                        scores[i][next_token] = a
                    else:
                        scores[i][next_token] = b + math.log(math.exp(a - b) - 1)
                self.hyps_logprobs[i] -= a
        scores = scores.log_softmax(-1)
        scores = torch.nan_to_num(scores, nan=-10000)
        self.importance_logits.append(scores.clone())
        return scores


def gen_samples(n_samples, model, batch, sample_tokens, sample_log_p, **args):
    batch_size = len(batch["input_ids"])
    logits, importance_logits = (
        [[] for _ in range(batch_size)],
        [[] for _ in range(batch_size)],
    )
    sequences = [[] for _ in range(batch_size)]
    with torch.no_grad():
        for k in range(n_samples):
            input_len = (
                batch["input_ids"].shape[1] if model.model_type == "CausalLM" else 0
            )
            logits_processor = EraseHypothesesLogitsProcessor(
                hyps_to_erase=[tokens[k] for tokens in sample_tokens],
                hyps_logprobs=[log_p[k] for log_p in sample_log_p],
                input_len=input_len,
            )
            out = model.generate(
                **batch,
                logits_processor=LogitsProcessorList([logits_processor]),
                **args
            )
            cur_logits = torch.stack(logits_processor.logits, dim=1).log_softmax(-1)
            cur_importance_logits = torch.stack(
                logits_processor.importance_logits, dim=1
            ).log_softmax(-1)
            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
                importance_logits[i].append(cur_importance_logits[i])
    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    logits = [l for sample_l in logits for l in sample_l]
    importance_logits = [l for sample_l in importance_logits for l in sample_l]
    return sequences, logits, importance_logits


class AdaptedSamplingGenerationCalculator(StatCalculator):
    def __init__(self, samples_n: int = 10):
        self.samples_n = samples_n
        super().__init__(
            [
                "adapted_sample_log_probs",
                "adapted_sample_log_probs_gen",
                "adapted_sample_tokens",
                "adapted_sample_texts",
            ],
            ["sample_log_probs", "sample_tokens", "sample_texts"],
        )

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, max_new_tokens: int = 100) -> Dict[str, np.ndarray]:
        sample_texts: List[List[str]] = \
            [samples[:(self.samples_n + 1) // 2] for samples in dependencies['sample_texts']]
        sample_tokens: List[List[List[int]]] = \
            [samples[:(self.samples_n + 1) // 2] for samples in dependencies['sample_tokens']]
        sample_log_p: List[List[float]] = \
            [samples_ll[:(self.samples_n + 1) // 2] for samples_ll in dependencies['sample_log_probs']]  # p
        sample_log_p_gen: List[List[float]] = \
            [samples_ll[:(self.samples_n + 1) // 2] for samples_ll in dependencies['sample_log_probs']]  # p'

        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        n_importance_samples = self.samples_n // 2
        importance_sequences, logits, importance_logits = gen_samples(
            n_importance_samples,
            model,
            batch,
            sample_tokens,
            sample_log_p,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            min_length=2,
            num_beams=1,
            do_sample=True)
        
        for i in range(len(importance_logits)):
            importance_log_prob, log_prob, toks = 0, 0, []
            inp_size = len(batch["input_ids"][int(i / n_importance_samples)])
            for j in range(len(importance_sequences[i]) - inp_size):
                cur_token = importance_sequences[i][j + inp_size].item()
                importance_log_prob += importance_logits[i][j][cur_token].item()
                log_prob += logits[i][j][cur_token].item()
                if cur_token == model.tokenizer.eos_token_id:
                    break
                toks.append(cur_token)
            sample_log_p_gen[int(i / n_importance_samples)].append(importance_log_prob)
            sample_log_p[int(i / n_importance_samples)].append(log_prob)
            sample_tokens[int(i / n_importance_samples)].append(toks)
            sample_texts[int(i / n_importance_samples)].append(
                model.tokenizer.decode(toks)
            )

        return {
            "adapted_sample_log_probs": sample_log_p,
            "adapted_sample_log_probs_gen": sample_log_p_gen,
            "adapted_sample_tokens": sample_tokens,
            "adapted_sample_texts": sample_texts,
        }
