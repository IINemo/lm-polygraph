import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, Model


class BlackboxSamplingGenerationCalculator(StatCalculator):
    def __init__(self, samples_n: int = 10):
        self.samples_n = samples_n
        super().__init__(['blackbox_sample_texts'], [])

    def __call__(self, dependencies: Dict[str, np.array],
                       texts: List[str],
                       model: BlackboxModel) -> Dict[str, np.ndarray]:

        if type(model) == BlackboxModel:
            samples = model.generate_texts(
                    input_texts=texts,
                    max_tokens=256,
                    temperature=model.parameters.temperature,
                    top_p=model.parameters.topp,
                    presence_penalty=model.parameters.presence_penalty,
                    repetition_penalty=model.parameters.repetition_penalty,
                    top_k=model.parameters.topk,
                    n=self.samples_n)
        else:
            samples = [[] for _ in range(len(texts))]
            out = model.generate_texts(
                    input_texts=texts,
                    max_length=256,
                    min_length=2,
                    do_sample=True,
                    num_beams=1,
                    temperature=model.parameters.temperature,
                    top_p=model.parameters.topp,
                    repetition_penalty=model.parameters.repetition_penalty,
                    top_k=model.parameters.topk,
                    num_return_sequences=self.samples_n)
            for i in range(len(texts)):
                for j in range(self.samples_n):
                    samples[i].append(out[i + j])
        
        return {
            'blackbox_sample_texts': samples,
        }


def gen_samples(n_samples, model, batch, **args):
    batch_size = len(batch['input_ids'])
    logits, sequences = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    with torch.no_grad():
        for k in range(n_samples):
            out = model.generate(
                **batch,
                **args
            )
            cur_logits = torch.stack(out.scores, dim=1).log_softmax(-1)
            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    logits = [l for sample_l in logits for l in sample_l]
    return sequences, logits


class SamplingGenerationCalculator(StatCalculator):
    def __init__(self, samples_n: int = 10):
        self.samples_n = samples_n
        super().__init__(['sample_log_probs', 'sample_tokens', 'sample_texts'], [])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel) -> Dict[
        str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        sequences, logits = gen_samples(
            self.samples_n, model, batch,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=256,
            min_length=2,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1)

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        if model.model_type == "Seq2SeqLM":
            sequences = [seq[1:] for seq in sequences]
        for i in range(len(logits)):
            log_prob, toks = 0, []
            inp_size = len(batch['input_ids'][int(i / self.samples_n)]) if model.model_type == "CausalLM" else 0
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
            'sample_log_probs': log_probs,
            'sample_tokens': tokens,
            'sample_texts': texts,
        }
