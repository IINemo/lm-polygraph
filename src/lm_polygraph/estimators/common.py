import torch
import numpy as np

from transformers import DebertaForSequenceClassification, DebertaTokenizer


class CommonDeberta:
    """
    Allows for the implementation of a singleton DeBERTa model which can be shared across
    different uncertainty estimation methods in the code.
    """

    def __init__(self, deberta_path="microsoft/deberta-large-mnli", device=None):
        """
        Parameters
        ----------
        deberta_path : str
            huggingface path of the pretrained DeBERTa (default 'microsoft/deberta-large-mnli')
        device : str
            device on which the computations will take place (default 'cuda:0' if available, else 'cpu').
        """
        self.deberta_path = deberta_path
        self.deberta = None
        self.deberta_tokenizer = None
        self.device = device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def to(self, device):
        self.device = device
        if self.deberta is not None:
            self.deberta.to(self.device)

    def setup(self):
        """
        Loads and prepares the DeBERTa model from the specified path.
        """
        if self.deberta is not None:
            return
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            self.deberta_path, problem_type="multi_label_classification"
        )
        self.deberta_tokenizer = DebertaTokenizer.from_pretrained(self.deberta_path)
        self.deberta.to(self.device)
        self.deberta.eval()


DEBERTA = CommonDeberta(device="cpu")


def _get_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pairs.append((lst[i], lst[j], i, j))
    return pairs


def _compute_Jaccard_score(lst):
    # device = DEBERTA.device
    jaccard_sim_mat = np.eye(len(lst))
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            set1 = set(lst[i].lower().split())
            set2 = set(lst[j].lower().split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            if union == 0:
                jaccard_score = 0
            else:
                jaccard_score = intersection / union
            jaccard_sim_mat[i, j] = jaccard_score
            jaccard_sim_mat[j, i] = jaccard_score

    return jaccard_sim_mat


def compute_sim_score(answers, affinity, similarity_score):
    return _compute_Jaccard_score(answers)
