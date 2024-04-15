import torch
import traceback
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


def _batch_tokens(tokens_list: List[List[int]], model: WhiteboxModel):
    token_tensors = [torch.tensor(t) for t in tokens_list]
    tokens = pad_sequence(
        token_tensors, batch_first=True, padding_value=model.tokenizer.pad_token_id
    )
    attn_mask = tokens != model.tokenizer.pad_token_id
    return {"input_ids": tokens, "attention_mask": attn_mask}


class ModelScoreCalculator(StatCalculator):
    def __init__(self, prompt: str = 'Paraphrase "{}": ', batch_size: int = 10):
        super().__init__(["model_rh"], ["greedy_tokens", "input_tokens"])
        self.batch_size = batch_size
        self.prompt = prompt

    def _score(
        self, model: WhiteboxModel, srcs: List[List[int]], tgts: List[List[int]]
    ) -> List[List[float]]:
        score_list = []
        for i in range(0, len(srcs), self.batch_size):
            src_list = srcs[i : i + self.batch_size]
            tgt_list = tgts[i : i + self.batch_size]
            try:
                with torch.no_grad():
                    encoded_src = _batch_tokens(
                        [s + t for s, t in zip(src_list, tgt_list)], model
                    )
                    src_tokens = encoded_src["input_ids"].to(model.device())
                    src_mask = encoded_src["attention_mask"].to(model.device())
                    if model.model_type == "CausalLM":
                        logits = model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                        ).logits
                    else:
                        encoded_src = _batch_tokens(src_list, model)
                        encoded_tgt = _batch_tokens(tgt_list, model)

                        src_tokens = encoded_src["input_ids"].to(model.device())
                        tgt_tokens = encoded_tgt["input_ids"].long().to(model.device())
                        src_mask = encoded_src["attention_mask"].to(model.device())

                        logits = model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            labels=tgt_tokens,
                        ).logits

                    for j, sample_logits in enumerate(logits):
                        score_list.append([])
                        for token_i, logits_i in enumerate(
                            range(len(logits) - len(tgt_list[j]) - 1, len(logits) - 1)
                        ):
                            score_list[-1].append(
                                sample_logits[logits_i, tgt_list[j][token_i]].item()
                            )
            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        # inp_tokens = dependencies["input_tokens"]
        preds = dependencies["greedy_tokens"]
        prompted_refs = model.tokenizer(
            [self.prompt.format(s) for s in dependencies["target_texts"]]
        )["input_ids"]

        scores = {"model_rh": self._score(model, prompted_refs, preds)}
        # scores["sh"] = self._score(model, inp_tokens, preds)
        # scores["hr"] = self._score(preds, refs)

        return scores
