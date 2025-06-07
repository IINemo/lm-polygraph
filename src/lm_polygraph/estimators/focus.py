import os
import numpy as np
import torch
import math
import dataclasses

from typing import Dict, Tuple, List

from spacy import Language

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


def calcu_idf(
    tokenizer_path, path, idf_dataset, trust_remote_code, idf_seed, idf_dataset_size
):
    """
    Calculate inverse document frequency (IDF) scores for each token using a Hugging Face tokenizer
    and dataset. Results are saved to disk for reuse.

    Args:
       tokenizer_path (str): Path to the tokenizer model.
       path (str): File path to save computed IDF values.
       idf_dataset (str): Hugging Face dataset identifier for IDF computation.
       trust_remote_code (bool): Whether to trust remote code when loading the dataset.
       idf_seed (int): Random seed for dataset shuffling.
       idf_dataset_size (int): Max number of documents to use (-1 for all).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = load_dataset(idf_dataset, trust_remote_code=trust_remote_code)
    data = [d for d in dataset["train"]]
    rng = random.Random(idf_seed)
    rng.shuffle(data)

    if (idf_dataset_size > 0) and (idf_dataset_size < len(data)):
        data = rng.sample(data, idf_dataset_size)

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


@dataclasses.dataclass
class IDFStats:
    """
    Container for IDF-related statistics and resources used in the Focus estimator.

    Attributes:
        token_idf (List): List of IDF values per token index.
        NER_type (List[str]): Named entity types considered important.
        pos_tag (List[str]): POS tags considered important.
        nlp (Language): Loaded spaCy NLP pipeline.
    """

    token_idf: List
    NER_type: List[str]
    pos_tag: List[str]
    nlp: Language


def load_idf(
    model_name: str,
    path: str,
    idf_dataset: str,
    trust_remote_code: bool,
    idf_seed: int,
    idf_dataset_size: int,
    spacy_path: str,
) -> IDFStats:
    """
    Load IDF statistics and spaCy model, computing IDF values if not already saved.

    Args:
        model_name (str): Tokenizer model name or path.
        path (str): Path to load or save the IDF file.
        idf_dataset (str): Dataset name used to calculate IDF.
        trust_remote_code (bool): Trust remote dataset loading code.
        idf_seed (int): Random seed for sampling.
        idf_dataset_size (int): Max number of samples to use for IDF.
        spacy_path (str): Name or path of the spaCy language model.

    Returns:
        IDFStats: Loaded or computed IDF statistics.
    """

    if not spacy.util.is_package(spacy_path):
        spacy.cli.download(spacy_path)
    if not os.path.exists(path):
        calcu_idf(
            model_name,
            path,
            idf_dataset,
            trust_remote_code,
            idf_seed,
            idf_dataset_size,
        )
    return IDFStats(
        token_idf=pickle.load(open(path, "rb")),
        NER_type=[
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
        ],
        pos_tag=["NOUN", "NUM", "PROPN"],
        nlp=spacy.load(spacy_path),
    )


def entropy2(p):
    """
    Compute the entropy of a probability distribution using base-2 logarithm.

    Args:
        p (array-like): Probability distribution.

    Returns:
        float: Entropy value.
    """

    p_torch = torch.tensor(p)
    return torch.sum(
        -torch.where(p_torch > 0, p_torch * p_torch.log2(), p_torch.new([0.0])),
        dim=-1,
    ).numpy()


def token_level_focus_scores(
    stats: Dict[str, np.ndarray],
    idf: IDFStats,
    p: float,
    gamma: float,
) -> Tuple[List, List]:
    """
    Compute token-level Focus uncertainty scores and keyword masks based on
    attention, IDF, and linguistic signals (NER, POS).

    Args:
        stats (Dict[str, np.ndarray]): Dictionary of model statistics including
            attention weights, log-probabilities, token IDs, and texts.
        idf (IDFStats): Precomputed IDF values and spaCy resources.
        p (float): Probability threshold for filtering low-confidence tokens.
        gamma (float): Smoothing parameter for context-aware uncertainty penalty.

    Returns:
        Tuple[List, List]: List of token-level uncertainty scores and keyword masks.
    """

    attention_weights = stats["attention_all"]
    greedy_log_probs = stats["greedy_log_probs"]
    greedy_tokens = stats["greedy_tokens"]
    greedy_texts = stats["greedy_texts"]
    tokenizer = stats["tokenizer"]
    loss_fct = NLLLoss(reduction="none")

    all_token_focus = []
    all_kw_mask = []
    for greedy_log_prob, attention_weight, greedy_token, greedy_text in zip(
        greedy_log_probs, attention_weights, greedy_tokens, greedy_texts
    ):
        sentence = idf.nlp(greedy_text)
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
                if span.text not in idf.NER_type and (
                    span.ent_type_ in idf.NER_type or span.pos_ in idf.pos_tag
                ):
                    kw_mask[token_index] = True
        except Exception as e:
            log.error(e, exc_info=True)
            log.info(decodings, sentence)
            pass

        prob = np.exp(greedy_log_prob)
        mask = prob < p
        # only focus on keywords like NER
        prob[mask] = 0
        if prob.shape[-1] > len(idf.token_idf):
            prob[:, : len(idf.token_idf)] = (
                prob[:, : len(idf.token_idf)] * idf.token_idf
            )
        else:
            prob = prob * idf.token_idf
        prob = prob / np.sum(prob, axis=-1, keepdims=True)
        entropy = np.exp2(entropy2(prob))

        ll = loss_fct(torch.log(torch.tensor(prob) + 1e-10), torch.tensor(greedy_token))
        hc = ll + entropy

        if not kw_mask.sum():
            all_token_focus.append([])
            all_kw_mask.append(kw_mask)
            continue
        # w(i,j) estimation and penalty estimation for a new hallucination score
        weight = np.zeros_like(attention_weight)
        weight[kw_mask] = attention_weight[kw_mask] / (
            np.sum(attention_weight[kw_mask], axis=1, keepdims=True) + 1e-6
        )
        token_focus = []
        for i, token_weights in enumerate(weight):
            ue = hc[i]
            if len(token_focus):
                ue += (
                    gamma
                    * (np.array(token_focus) * token_weights[: len(token_focus)]).sum()
                )
            token_focus.append(ue)
        all_token_focus.append(token_focus)
        all_kw_mask.append(kw_mask)

    return all_token_focus, all_kw_mask


class Focus(Estimator):
    """
    Implements the Focus uncertainty estimator as described in:
    "Hallucination Detection in Neural Text Generation via Focused Uncertainty Estimation"
    (https://arxiv.org/abs/2311.13230).

    Args:
        gamma (float): Context penalty coefficient that controls influence of surrounding tokens.
        p (float): Probability threshold below which token predictions are masked out.
        model_name (str): Hugging Face model name or path to the tokenizer.
        path (str): Path to save or load precomputed IDF values.
        idf_dataset (str): Dataset name used to calculate IDF values.
        trust_remote_code (bool): Whether to allow loading of custom dataset scripts.
        idf_seed (int): Random seed used to shuffle or sample dataset.
        idf_dataset_size (int): Number of examples to use for IDF computation (-1 for all).
        spacy_path (str): Name or path of spaCy language model to use for POS/NER parsing.
    """

    def __init__(
        self,
        gamma: float,
        p: float,
        model_name: str,
        path: str,
        idf_dataset: str,
        trust_remote_code: bool,
        idf_seed: int,
        idf_dataset_size: int,
        spacy_path: str,
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

        self.p = p
        self.gamma = gamma
        self.idf_stats = load_idf(
            model_name,
            path,
            idf_dataset,
            trust_remote_code,
            idf_seed,
            idf_dataset_size,
            spacy_path,
        )

    def __str__(self):
        return f"Focus (gamma={self.gamma})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate token-level Focus scores into a sentence-level hallucination score.

        Args:
            stats (Dict[str, np.ndarray]): Dictionary of generation statistics including
                attention maps, token probabilities, and decoded text.

        Returns:
            np.ndarray: Sentence-level Focus uncertainty scores.
        """
        all_token_focus, all_kw_mask = token_level_focus_scores(
            stats,
            self.idf_stats,
            self.p,
            self.gamma,
        )

        focus_ue = []
        for token_focus, kw_mask in zip(all_token_focus, all_kw_mask):
            focus_ue.append(np.mean(np.array(token_focus)[kw_mask]))

        return np.array(focus_ue)
