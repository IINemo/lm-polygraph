# %%
import traceback
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

from .stat_calculator import StatCalculator

from lm_polygraph.utils.model import WhiteboxModel


class BartScoreCalculator(StatCalculator):
    def __init__(
        self, device=None, max_length=256, checkpoint="facebook/bart-large-cnn"
    ):
        super().__init__(["rh"], ["greedy_tokens", "input_tokens"])
        self.max_length = max_length
        self.checkpoint = checkpoint
        self.device = device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self.loss_fct = None
        self.lsm = None

    def _setup(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(self.checkpoint)
        self.model.eval()
        self.model.to(self.device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """Load model from paraphrase finetuning"""
        if path is None:
            path = "models/bart.pth"
        if self.model is None:
            self._setup()
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """Score a batch of examples"""
        if self.model is None:
            self._setup()
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        if self.model is None:
            self._setup()
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """Test"""
        if self.model is None:
            self._setup()
        src_list = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]

        tgt_list = ["That's stupid.", "What's the problem?", "He is trustworthy."]

        print(self.score(src_list, tgt_list, batch_size))

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if self.model is None:
            self._setup()
        srcs, tgts = dependencies["greedy_texts"], dependencies["target_texts"]
        self.device = model.device()
        self.model.to(self.device)

        scores = {"rh": self.score(srcs, tgts)}

        return scores
