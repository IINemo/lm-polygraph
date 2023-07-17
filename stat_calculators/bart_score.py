# %%
import traceback
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

from stat_calculators.stat_calculator import StatCalculator
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.model import Model

class BartScoreCalculator(StatCalculator):
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        super().__init__(['rh'],
                         ['greedy_tokens', 'input_tokens'])
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
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
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))
        
    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        srcs, tgts = dependencies['greedy_texts'], dependencies['target_texts']
        self.device = model.device()
        self.model.to(self.device)
                
        scores = {
            'rh': self.score(srcs, tgts)
        }

        return scores

# def _batch_tokens(tokens_list: List[List[int]], model: Model, max_len: int = None):
#     if max_len is None:
#         max_len = max(len(tokens) for tokens in tokens_list)
#     tokens = torch.from_numpy(pad_sequences(
#         tokens_list, maxlen=max_len, padding='pre', truncating='pre',
#         value=model.tokenizer.pad_token_id))
#     attn_mask = (tokens != model.tokenizer.pad_token_id)
#     return {'input_ids': tokens, 'attention_mask': attn_mask}


# class BartScoreCalculator(StatCalculator):
#     def __init__(self, prompt: str = 'Paraphrase "{}": ', batch_size: int = 10):
#         super().__init__(['rh'],
#                          ['greedy_tokens', 'input_tokens'])
#         self.batch_size = batch_size
#         self.prompt = prompt

#     def _score(self, model: Model, srcs: List[List[int]], tgts: List[List[int]]) -> List[List[float]]:
#         score_list = []
#         for i in range(0, len(srcs), self.batch_size):
#             src_list = srcs[i:i + self.batch_size]
#             tgt_list = tgts[i:i + self.batch_size]
#             try:
#                 with torch.no_grad():
#                     encoded_src = _batch_tokens([s + t for s, t in zip(src_list, tgt_list)], model)
#                     src_tokens = encoded_src['input_ids'].to(model.device())
#                     src_mask = encoded_src['attention_mask'].to(model.device())
#                     if model.model_type == "CausalLM":
#                         logits = model.model(
#                             input_ids=src_tokens,
#                             attention_mask=src_mask,
#                         ).logits
#                     else:
#                         max_len = max(max(len(s) for s in src_list), max(len(t) for t in tgt_list))
#                         encoded_src = _batch_tokens(src_list, model, max_len=max_len)
#                         encoded_tgt = _batch_tokens(tgt_list, model, max_len=max_len)
                        
#                         src_tokens = encoded_src['input_ids'].to(model.device())
#                         tgt_tokens = encoded_tgt['input_ids'].long().to(model.device())
#                         src_mask = encoded_src['attention_mask'].to(model.device())
                        
#                         logits = model.model(
#                             input_ids=src_tokens,
#                             attention_mask=src_mask,
#                             labels=tgt_tokens
#                         ).logits
                        
#                     for j, sample_logits in enumerate(logits):
#                         score_list.append([])
#                         for token_i, logits_i in enumerate(range(len(logits) - len(tgt_list[j]) - 1, len(logits) - 1)):
#                             score_list[-1].append(sample_logits[logits_i, tgt_list[j][token_i]].item())
#             except RuntimeError:
#                 traceback.print_exc()
#                 print(f'source: {src_list}')
#                 print(f'target: {tgt_list}')
#                 exit(0)
#         return score_list

#     def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
#         preds, inp_tokens = dependencies['greedy_tokens'], dependencies['input_tokens']
#         prompted_refs = model.tokenizer(
#             [self.prompt.format(s) for s in dependencies['target_texts']])['input_ids']

#         scores = {
#             'rh': self._score(model, prompted_refs, preds)
#         }
#         # scores["sh"] = self._score(model, inp_tokens, preds)
#         # scores["hr"] = self._score(preds, refs)

#         return scores
