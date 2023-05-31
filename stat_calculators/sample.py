import torch
import numpy as np

from typing import Dict, List

from stat_calculators.stat_calculator import StatCalculator
from utils.model import Model


def gen_samples(n_samples, model, batch, **args):
    batch_size = len(batch['input_ids'])
    logits, sequences = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    with torch.no_grad():
        for k in range(n_samples):
            out = model.model.generate(
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

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        sequences, logits = gen_samples(
            self.samples_n, model, batch,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=256,
            min_length=2,
            do_sample=True,
            num_return_sequences=1)

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        for i in range(len(logits)):
            log_prob, toks = 0, []
            inp_size = len(batch['input_ids'][int(i / self.samples_n)])
            for j in range(len(sequences[i]) - inp_size):
                cur_token = sequences[i][j + inp_size].item()
                log_prob += max(logits[i][j][cur_token].item(), -10)
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
