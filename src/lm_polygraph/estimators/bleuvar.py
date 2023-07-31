import numpy as np
from tqdm import tqdm
from typing import Dict, List
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import corpus_bleu

from .estimator import Estimator


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    smoothed_p_n = []
    for i, p_i in enumerate(p_n, start=1):
        # Smoothing is not applied for unigrams
        if i > 1:
            # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0
            if hyp_len < i:
                assert p_i.denominator == 1
                smoothed_p_n.append(1)
            # Otherwise apply smoothing
            else:
                smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                smoothed_p_n.append(smoothed_p_i)
        else:
            smoothed_p_n.append(p_i)
    return smoothed_p_n

def pair_bleu(references, prediction):
    """
    Compute the bleu score between two given texts.
    A smoothing function is used to avoid zero scores when
    there are no common higher order n-grams between the
    texts.
    """
    tok_ref = [word_tokenize(s) for s in sent_tokenize(references)]
    tok_pred = [word_tokenize(s) for s in sent_tokenize(prediction)]
    score = 0
    for c_cent in tok_pred:
        try:
            score += corpus_bleu(
                [tok_ref], [c_cent], smoothing_function=smoothing_function
            )
        except KeyError:
            score = 0.0
    try:
        score /= len(tok_pred)
    except ZeroDivisionError:
        score = 0.0

    return score


class BLEUVar(Estimator):
    def __init__(self):
        super().__init__(['bleuvar'], 'sequence')

    def __str__(self):
        return 'BLEUVar'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        hypotheses: List[List[str]] = stats['ensemble_hypotheses']
        bleu_vars = []
        for inst_hypotheses in tqdm(hypotheses, desc='Calculating BLEUVar scores...'):
            n = len(inst_hypotheses)
            bleu_scores = np.zeros((n, n), dtype=float)
            min_bleuvar = float("inf")
            for j, dec_j in enumerate(inst_hypotheses):
                for k in range(j + 1, n):
                    dec_k = inst_hypotheses[k]
                    jk_bleu = pair_bleu(dec_j, dec_k)
                    kj_bleu = pair_bleu(dec_k, dec_j)

                    bleu_scores[j, k] = 1 - jk_bleu
                    bleu_scores[k, j] = 1 - kj_bleu

                mu_bleuvar = np.sum(bleu_scores[j, :]) + np.sum(bleu_scores[:, j])
                if mu_bleuvar < min_bleuvar:
                    min_bleuvar = mu_bleuvar

            bleu_var = (bleu_scores ** 2).sum() / (n * (n - 1))
            bleu_vars.append(bleu_var)

        return np.array(bleu_vars)
