import os
import numpy as np
import torch
import math

from typing import Dict
from .estimator import Estimator

import pickle
from tqdm import tqdm
from torch.nn import NLLLoss

from datasets import load_dataset
from collections import defaultdict
import random
import logging

from transformers import AutoTokenizer
import spacy


log = logging.getLogger(__name__)


def calcu_idf(tokenizer_path, path, idf_dataset, trust_remote_code, idf_seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = load_dataset(idf_dataset, trust_remote_code=trust_remote_code)
    data = [d for d in dataset["train"]]
    random.seed(idf_seed)
    random.shuffle(data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    document_frequency = defaultdict(int)
    offset = 1 if "facebook" in tokenizer_path else 0
    for doc in tqdm(data):
        tokenized_doc = tokenizer(doc["text"])["input_ids"][offset:]
        unique_tokens = set(tokenized_doc)
        for token in unique_tokens:
            document_frequency[token] += 1
    total_documents = len(data)
    pickle.dump(
        np.array(
            [
                math.log(total_documents / (document_frequency[i] + 1))
                for i in range(len(tokenizer.vocab))
            ]
        ),
        open(path, "wb"),
    )


class Focus(Estimator):
    def __init__(
        self,
        gamma: float = 0.9,
        p: float = 0.01,
        model_name: str = "meta-llama/Llama-3.2-1B",
        path: str = None,
        idf_dataset: str = "togethercomputer/RedPajama-Data-1T-Sample",
        trust_remote_code: bool = True,
        idf_seed: int = 42,
        spacy_path: str = "en_core_web_sm",
    ):
        super().__init__(
            [
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_texts",
                "attention_all",
                "tokenizer",
            ],
            "sequence",
        )
        self.path = path or f"../focus_data/token_idf_{model_name.split('/')[-1]}.pkl"
        spacy.cli.download(spacy_path)
        if not os.path.exists(self.path):
            calcu_idf(model_name, self.path, idf_dataset, trust_remote_code, idf_seed)
        self.token_idf = pickle.load(open(self.path, "rb"))
        self.NER_type = [
            "PERSON",
            "DATE",
            "ORG",
            "GPE",
            "NORP",
            "ORDINAL",
            "PRODUCT",
            "CARDINAL",
            "LOC",
            "FAC",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
        ]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]
        self.start_token_idx = 999999
        self.p = p
        self.gamma = gamma
        self.nlp = spacy.load(spacy_path)

    def __str__(self):
        return f"Focus (gamma={self.gamma})"

    def entropy(self, p):
        p_torch = torch.tensor(p)
        return torch.sum(
            -torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])),
            dim=-1,
        ).numpy()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        attention_weights = stats["attention_all"]
        greedy_log_probs = stats["greedy_log_probs"]
        greedy_tokens = stats["greedy_tokens"]
        greedy_texts = stats["greedy_texts"]
        tokenizer = stats["tokenizer"]
        loss_fct = NLLLoss(reduction="none")

        focus_ue = []
        for greedy_log_prob, attention_weight, greedy_token, greedy_text in zip(
            greedy_log_probs, attention_weights, greedy_tokens, greedy_texts
        ):
            sentence = self.nlp(greedy_text)
            decodings = tokenizer.batch_decode(greedy_token, skip_special_tokens=True)
            span_index = 0
            kw_mask = np.zeros_like(greedy_token, dtype=bool)
            try:
                for token_index, token in enumerate(decodings):
                    while (token.strip() not in sentence[span_index].text) and (
                        sentence[span_index].text not in token.strip()
                    ):
                        span_index += 1
                    span = sentence[span_index]
                    if span.text not in self.NER_type and (
                        span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag
                    ):
                        kw_mask[token_index] = True
            except Exception as e:
                log.error(e, exc_info=True)
                log.info(decodings, sentence)
                pass

            prob = np.exp(greedy_log_prob)
            mask = prob < self.p
            # only focus on keywords like NER
            prob[mask] = 0
            if prob.shape[-1] > len(self.token_idf):
                prob[:, : len(self.token_idf)] = (
                    prob[:, : len(self.token_idf)] * self.token_idf
                )
            else:
                prob = prob * self.token_idf
            prob = prob / np.sum(prob, axis=-1, keepdims=True)
            entropy = np.exp2(self.entropy(prob))

            ll = loss_fct(
                torch.log(torch.tensor(prob) + 1e-10), torch.tensor(greedy_token)
            )
            hc = ll + entropy

            if not kw_mask.sum():
                focus_ue.append(hc.mean())
                continue
            # w(i,j) estimation and penalty estimation for a new gallucination score
            weight = attention_weight[kw_mask] / (
                np.sum(attention_weight[kw_mask], axis=1, keepdims=True) + 1e-6
            )
            weight = np.zeros_like(attention_weight)
            weight[kw_mask] = attention_weight[kw_mask] / (
                np.sum(attention_weight[kw_mask], axis=1, keepdims=True) + 1e-6
            )
            token_focus = []
            for i, token_weights in enumerate(weight):
                ue = hc[i]
                if len(token_focus):
                    ue += (
                        self.gamma
                        * (
                            np.array(token_focus) * token_weights[: len(token_focus)]
                        ).sum()
                    )
                token_focus.append(ue)
            focus_ue.append(np.mean(np.array(token_focus)[kw_mask]))

        return np.array(focus_ue)


class FocusClaim(Estimator):
    def __init__(
        self,
        gamma: float = 0.9,
        p: float = 0.01,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        path: str = None,
        idf_dataset: str = "togethercomputer/RedPajama-Data-1T-Sample",
        trust_remote_code: bool = True,
        idf_seed: int = 42,
        spacy_path: str = "en_core_web_sm",
    ):
        super().__init__(
            [
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_texts",
                "claims",
                "attention_all",
                "tokenizer",
            ],
            "claim",
        )
        self.path = path or f"../focus_data/token_idf_{model_name.split('/')[-1]}.pkl"
        if not os.path.exists(self.path):
            calcu_idf(model_name, self.path, idf_dataset, trust_remote_code, idf_seed)
        self.token_idf = pickle.load(open(self.path, "rb"))
        self.NER_type = [
            "PERSON",
            "DATE",
            "ORG",
            "GPE",
            "NORP",
            "ORDINAL",
            "PRODUCT",
            "CARDINAL",
            "LOC",
            "FAC",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
        ]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]
        self.p = p
        self.gamma = gamma
        self.nlp = spacy.load(spacy_path)

    def __str__(self):
        return f"FocusClaim (gamma={self.gamma})"

    def entropy(self, p):
        p_torch = torch.tensor(p)
        return torch.sum(
            -torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])),
            dim=-1,
        ).numpy()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        attention_weights = stats["attention_all"]
        greedy_log_probs = stats["greedy_log_probs"]
        greedy_tokens = stats["greedy_tokens"]
        greedy_texts = stats["greedy_texts"]
        tokenizer = stats["tokenizer"]
        claims = stats["claims"]
        loss_fct = NLLLoss(reduction="none")

        focus_ue = []
        for (
            greedy_log_prob,
            attention_weight,
            greedy_token,
            greedy_text,
            claims_i,
        ) in zip(
            greedy_log_probs, attention_weights, greedy_tokens, greedy_texts, claims
        ):
            sentence = self.nlp(greedy_text)
            decodings = tokenizer.batch_decode(greedy_token, skip_special_tokens=True)
            span_index = 0
            kw_mask = np.zeros_like(greedy_token, dtype=bool)
            try:
                for token_index, token in enumerate(decodings):
                    while (token.strip() not in sentence[span_index].text) and (
                        sentence[span_index].text not in token.strip()
                    ):
                        span_index += 1
                    span = sentence[span_index]
                    if span.text not in self.NER_type and (
                        span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag
                    ):
                        kw_mask[token_index] = True
            except Exception as e:
                log.error(e, exc_info=True)
                log.info(decodings, sentence)
                pass

            prob = np.exp(greedy_log_prob)
            mask = prob < self.p
            prob[mask] = 0
            if prob.shape[-1] > len(self.token_idf):
                prob[:, : len(self.token_idf)] = (
                    prob[:, : len(self.token_idf)] * self.token_idf
                )
            else:
                prob = prob * self.token_idf
            prob = prob / np.sum(prob, axis=-1, keepdims=True)
            entropy = np.exp2(self.entropy(prob))
            ll = loss_fct(
                torch.log(torch.tensor(prob) + 1e-10), torch.tensor(greedy_token)
            )
            hc = ll + entropy

            if not kw_mask.sum():
                kw_mask = np.ones_like(greedy_token, dtype=bool)

            kw_mask = kw_mask.flatten()

            weight = np.zeros_like(attention_weight)
            weight[kw_mask] = attention_weight[kw_mask] / (
                np.sum(attention_weight[kw_mask], axis=1, keepdims=True) + 1e-6
            )

            token_focus = []
            for i, token_weights in enumerate(weight):
                ue = hc[i]
                if len(token_focus):
                    ue += (
                        self.gamma
                        * (
                            np.array(token_focus) * token_weights[: len(token_focus)]
                        ).sum()
                    )
                token_focus.append(ue)
            token_focus = np.array(token_focus)

            focus_ue.append([])
            for claim in claims_i:
                tokens = np.array(claim.aligned_token_ids)
                claim_p_i = token_focus[tokens]
                focus_ue[-1].append(claim_p_i.mean())
        return focus_ue
