import os
import numpy as np
import torch
import math

from typing import Dict

from .estimator import Estimator

import pickle
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
import random

from transformers import AutoTokenizer

import spacy
nlp = spacy.load('en_core_web_sm')


#calculating tokens idf on 1M documents from RedPajama dataset 
def calcu_idf(tokenizer_path, path):
    print(tokenizer_path)
    filename = "/tmp/not_exist/filenames.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    data = [d for d in dataset["train"]]
    random.seed(42)
    random.shuffle(data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    document_frequency = defaultdict(int)
    offset = 1 if 'facebook' in tokenizer_path else 0
    for doc in tqdm(data):
        tokenized_doc = tokenizer(doc["text"])["input_ids"][offset:]
        unique_tokens = set(tokenized_doc)
        for token in unique_tokens:
            document_frequency[token] += 1
    total_documents = len(data)
    print(np.array([math.log(total_documents / (document_frequency[i] + 1)) for i in range(len(tokenizer.vocab))]))
    pickle.dump(np.array([math.log(total_documents / (document_frequency[i] + 1)) for i in range(len(tokenizer.vocab))]), open(path, "wb"))


class Focus(Estimator):
    def __init__(
        self,
        gamma: float = 0.9,
        p: float = 0.01,
        model_name: str = "meta-llama/Llama-3.2-1B",
    ):
        super().__init__([
                            "greedy_log_probs",
                            "greedy_tokens", 
                            "greedy_texts",
                            "attention_all",
                            "tokenizer",
                         ], 
                         "sequence")
        #path where idf scores are stored 
        path = f"../../../focus_data/token_idf_{model_name.split('/')[-1]}.pkl"
        if not os.path.exists(path):
            calcu_idf(model_name, path)
        self.token_idf = pickle.load(open(path, "rb"))
        self.NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]
        self.start_token_idx = 999999
        self.p = p
        self.gamma = gamma
        
    def __str__(self):
        return f"Focus (gamma={self.gamma})"

    def entropy(self, p):
        p_torch = torch.tensor(p)
        return torch.sum(-torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])), dim=-1).numpy()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        attention_weights = stats["attention_all"]
        greedy_log_probs = stats["greedy_log_probs"]
        greedy_tokens = stats["greedy_tokens"]
        greedy_texts = stats["greedy_texts"]
        tokenizer = stats["tokenizer"]

        focus_ue = []
        for greedy_log_prob, attention_weight, greedy_token, greedy_text in zip(greedy_log_probs, attention_weights, greedy_tokens, greedy_texts):

            sentence = nlp(greedy_text)
            decodings = tokenizer.batch_decode(greedy_token, skip_special_tokens=True)
            span_index = 0
            kw_mask = np.zeros_like(greedy_token, dtype=bool)
            try:
                for token_index, token in enumerate(decodings):
                    while (token.strip() not in sentence[span_index].text) and (sentence[span_index].text not in token.strip()):
                        span_index += 1
                    span = sentence[span_index]
                    if span.text not in self.NER_type and (span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag):
                        kw_mask[token_index] = True
            except:
                print("Error with indexing")
                print(decodings, sentence)
                pass

            kw_mask =  kw_mask.flatten()
            
            prob = np.exp(greedy_log_prob)
            mask = prob < self.p
            # only focus on keywords like NER 
            prob[mask] = 0
            if prob.shape[-1] > len(self.token_idf):
                prob[:, :len(self.token_idf)] = prob[:, :len(self.token_idf)] * self.token_idf
            else:
                prob = prob * self.token_idf
            prob = prob / np.sum(prob, axis=-1, keepdims=True)
            entropy = np.exp2(self.entropy(prob))
            ll = np.log(np.array([prob[j, greedy_token[j]] for j in range(len(prob))]))
            hc = ll + entropy

            if not kw_mask.sum():
                focus_ue.append(hc.mean())
                continue
            # w(i,j) estimation and penalty estimation for a new gallucination score
            weight = attention_weight[kw_mask] / (np.sum(attention_weight[kw_mask], axis=1, keepdims=True)+1e-6)
            penalty = self.gamma * (hc[None, :] * weight).sum(0)

            focus = (hc + penalty).mean() 
            focus_ue.append(focus)
        
        return np.array(focus_ue)


class FocusClaim(Estimator):
    def __init__(
        self,
        gamma: float = 0.9,
        p: float = 0.01,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
    ):
        super().__init__([
                            "greedy_log_probs", 
                            "greedy_tokens", 
                            "greedy_texts", 
                            "claims",
                            "attention_all",
                            "tokenizer",
                         ], 
                         "claim",
                        )
        path = f"focus_data/token_idf_{model_name.split('/')[-1]}.pkl"
        if not os.path.exists(path):
            calcu_idf(model_name, path)
        self.token_idf = pickle.load(open(path, "rb"))
        self.NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]
        self.p = p
        self.gamma = gamma
        
    def __str__(self):
        return f"FocusClaim (gamma={self.gamma})"

    def entropy(self, p):
        p_torch = torch.tensor(p)
        return torch.sum(-torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])), dim=-1).numpy()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        attention_weights = stats["attention_all"]
        greedy_log_probs = stats["greedy_log_probs"]
        greedy_tokens = stats["greedy_tokens"]
        greedy_texts = stats["greedy_texts"]
        tokenizer = stats["tokenizer"]
        claims = stats["claims"]
        
        focus_ue = []
        for greedy_log_prob, attention_weight, greedy_token, greedy_text, claims_i in zip(greedy_log_probs, attention_weights, 
                                                                                          greedy_tokens, greedy_texts, claims):

            sentence = nlp(greedy_text)
            decodings = tokenizer.batch_decode(greedy_token, skip_special_tokens=True)
            span_index = 0
            kw_mask = np.zeros_like(greedy_token, dtype=bool)
            try:
                for token_index, token in enumerate(decodings):
                    while (token.strip() not in sentence[span_index].text) and (sentence[span_index].text not in token.strip()):
                        span_index += 1
                    span = sentence[span_index]
                    if span.text not in self.NER_type and (span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag):
                        kw_mask[token_index] = True
            except:
                print("Error with indexing")
                print(decodings, sentence)
                pass
            
            prob = np.exp(greedy_log_prob)
            mask = prob < self.p
            prob[mask] = 0
            if prob.shape[-1] > len(self.token_idf):
                prob[:, :len(self.token_idf)] = prob[:, :len(self.token_idf)] * self.token_idf
            else:
                prob = prob * self.token_idf
            prob = prob / np.sum(prob, axis=-1, keepdims=True)
            entropy = np.exp2(self.entropy(prob))
            ll = np.log(np.array([prob[j, greedy_token[j]] for j in range(len(prob))]))
            hc = ll + entropy

            if not kw_mask.sum():
                kw_mask = np.ones_like(greedy_token, dtype=bool)

            kw_mask =  kw_mask.flatten()

            weight = attention_weight[kw_mask] / (np.sum(attention_weight[kw_mask], axis=1, keepdims=True)+1e-6)
            penalty = self.gamma * (hc[None, :] * weight).sum(0)

            focus = (hc + penalty) 
            focus_ue.append([])
            for claim in claims_i:
                tokens = np.array(claim.aligned_token_ids)
                claim_p_i = focus[tokens]
                focus_ue[-1].append(claim_p_i.mean())
                
        return focus_ue
