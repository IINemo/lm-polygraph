import torch
import torch.nn as nn
import numpy as np

from transformers import DebertaForSequenceClassification, DebertaTokenizer


class CommonDeberta:
    def __init__(self, deberta_path='microsoft/deberta-large-mnli', device=None):
        self.deberta_path = deberta_path
        self.deberta = None
        self.deberta_tokenizer = None
        self.device = device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def setup(self):
        if self.deberta is not None:
            return
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            self.deberta_path, problem_type="multi_label_classification")
        self.deberta_tokenizer = DebertaTokenizer.from_pretrained(self.deberta_path)
        self.deberta.to(self.device)
        self.deberta.eval()


DEBERTA = CommonDeberta()


def _get_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pairs.append((lst[i], lst[j], i, j))
    return pairs


def _compute_Jaccard_score(lst):
    #device = DEBERTA.device 
    jaccard_sim_mat = np.eye(len(lst))
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            set1 = set(lst[i].lower().split())
            set2 = set(lst[j].lower().split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_score = intersection / union
            jaccard_sim_mat[i, j] = jaccard_score
            jaccard_sim_mat[j, i] = jaccard_score

    return jaccard_sim_mat


def _compute_adjaency_mat(answers, affinity):
    W = np.eye(len(answers))
    pairs = _get_pairs(answers)
    device = DEBERTA.device 

    softmax = nn.Softmax(dim=1)

    for (sentence_1, sentence_2, i, j) in pairs:
        # Tokenize input sentences
        encoded_input_forward = DEBERTA.deberta_tokenizer(sentence_1, sentence_2, return_tensors='pt').to(device)
        encoded_input_backward = DEBERTA.deberta_tokenizer(sentence_2, sentence_1, return_tensors='pt').to(device)

        logits_forward = DEBERTA.deberta(**encoded_input_forward).logits.detach().to(device)
        logits_backward = DEBERTA.deberta(**encoded_input_backward).logits.detach()

        probs_forward = softmax(logits_forward).to(device)
        probs_backward = softmax(logits_backward).to(device)

        a_nli_entail_forward = probs_forward[0][2]
        a_nli_entail_backward = probs_backward[0][2]

        a_nli_contra_forward = 1 - probs_forward[0][0]
        a_nli_contra_backward = 1 - probs_backward[0][0]

        if affinity == "entail":
            _w = (a_nli_entail_forward.item() + a_nli_entail_backward.item()) / 2
        elif affinity == "contra":
            _w = (a_nli_contra_forward.item() + a_nli_contra_backward.item()) / 2

        W[i, j] = _w
        W[j, i] = _w

    return W


def compute_sim_score(answers, affinity, similarity_score):
    if similarity_score == "NLI_score":
        return _compute_adjaency_mat(answers, affinity)
    elif similarity_score == "Jaccard_score":
        return _compute_Jaccard_score(answers)
