# imports
import sys
import os
import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Literal



from .estimator import Estimator
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from scipy.linalg import eigh
import torch.nn as nn
softmax = nn.Softmax(dim=1)


class BlackBoxUE(Estimator):
    def __init__(
            self, 
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            order: Literal["entail", "contra"] = "entail", # relevant for NLI score case
            uncertainty_method: Literal["Num_Sem_Sets", "Eig_Val_Laplacian", "Deg_Mat", "Eccentricity"] = "Eig_Val_Laplacian",
            deberta_path: str = 'microsoft/deberta-large-mnli',
            batch_size: int = 10,
            verbose: bool = False,
            epsilon: float = 1e-13
    ):
        """
        Initialize BlackBoxUE uncertainty method.

        Parameters:
            similarity_score (str): The argument to be processed. Possible values are:
                - 'NLI_score': As a similarity score NLI score is used.
                - 'Jaccard_score': As a similarity Jaccard score between responces is used.
            uncertainty_method (str): The argument to be processed. Possible values are:
                - 'Num_Sem_Sets': A number of semantic sets in response (higher = bigger uncertainty).
                - 'Eig_Val_Laplacian': (Due to the Theorem) A continuous analogue to the number of semantic sets (higher = bigger uncertainty).
                - 'Deg_Mat': Elements on diagonal of matrix D are sums of similarities between the particular number (position in matrix) and other answers. Thus, it is an average pairwise distance (less = more confident because distance between answers is smaller or higher = bigger uncertainty).
                - 'Eccentricity': It is a frobenious norm (euclidian norm) between all eigenvectors that are informative embeddings of graph Laplacian (lower this value -> answers are closer in terms of euclidian distance between embeddings = eigenvectors or higher = bigger uncertainty).
            order (str): The argument to be processed. Possible values are. Relevant for the case of NLI similarity score: 
                - 'forward': Compute entailment probability between response_1 and response_2 as p(response_1 -> response_2).
                - 'backward': Compute entailment probability between response_1 and response_2 as p(response_2 -> response_1).
        """
        super().__init__(['sample_texts'], 'sequence')
        self.similarity_score = similarity_score
        self.uncertainty_method = uncertainty_method
        self.batch_size = batch_size
        if self.similarity_score == "NLI_score":
            self.init_deberta(deberta_path)
        self.order = order
        self.verbose = verbose
        self.epsilon = epsilon
        

    def init_deberta(self, deberta_path):
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            deberta_path, problem_type="multi_label_classification")
        self.deberta_tokenizer = DebertaTokenizer.from_pretrained(deberta_path)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.deberta.to(self.device)
        self.deberta.eval()
    
    def __str__(self):
        return f'{self.uncertainty_method}_{self.similarity_score}'

    def get_pairs(self, lst):
        pairs = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                pairs.append((lst[i], lst[j], i, j))
        return pairs

    def get_pairs_semsets(self, lst):
        pairs = []
        for i in range(len(lst)-1):
                pairs.append((lst[i], lst[i+1]))
        return pairs

    def compute_Jaccard_score(self, lst):
        jaccard_sim_mat = np.eye(len(lst)) * self.epsilon
        scores = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                set1 = set(lst[i].lower().split())
                set2 = set(lst[j].lower().split())
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard_score = intersection / union
                jaccard_sim_mat[i, j] = jaccard_score

        return jaccard_sim_mat
    

    def compute_adjaency_mat(self, answers):
        W = np.eye(len(answers))
        pairs = self.get_pairs(answers)

        for (sentence_1, sentence_2, i ,j) in pairs:
            # Tokenize input sentences
            encoded_input_forward = self.deberta_tokenizer(sentence_1, sentence_2, return_tensors='pt')
            encoded_input_backward = self.deberta_tokenizer(sentence_2, sentence_1, return_tensors='pt')

            logits_forward = self.deberta(**encoded_input_forward).logits.detach()
            logits_backward = self.deberta(**encoded_input_backward).logits.detach()

            probs_forward = softmax(logits_forward)
            probs_backward = softmax(logits_backward)

            a_nli_entail_forward = probs_forward[0][2]
            a_nli_entail_backward = probs_backward[0][2]

            a_nli_contra_forward = 1 - probs_forward[0][0]
            a_nli_contra_backward = 1 - probs_backward[0][0]

            if self.order == "entail":
                w = (a_nli_entail_forward + a_nli_entail_backward) / 2
            elif self.order == "contra":
                w = (a_nli_contra_forward + a_nli_contra_backward) / 2
            
            W[i, j] = w
            W[j, i] = w

        return W
    
    def U_NumSemSets(self, answers):
        
        lst = self.get_pairs_semsets(answers)
        # basically we have only 1 semantic set
        num_sets = 1

        # we iterate over responces and incerase num_sets if the NLI condition is fulfilled 
        for (sentence_1, sentence_2) in lst:
            # Tokenize input sentences
            encoded_input_forward = self.deberta_tokenizer(sentence_1, sentence_2, return_tensors='pt')
            encoded_input_backward = self.deberta_tokenizer(sentence_2, sentence_1, return_tensors='pt')

            logits_forward = self.deberta(**encoded_input_forward).logits.detach()
            logits_backward = self.deberta(**encoded_input_backward).logits.detach()

            probs_forward = softmax(logits_forward)
            probs_backward = softmax(logits_backward)

            p_entail_forward = probs_forward[0][2]
            p_entail_backward = probs_backward[0][2]

            p_contra_forward = probs_forward[0][0]
            p_contra_backward = probs_backward[0][0]

            if (p_entail_forward > p_contra_forward) & (p_entail_backward > p_contra_backward):
                pass
            else:
                num_sets += 1
        
        return num_sets

    def U_EigVal_Laplacian(self, answers):
        if self.similarity_score == "NLI_score":
            W = self.compute_adjaency_mat(answers)
        elif self.similarity_score == "Jaccard_score":
            W = self.compute_Jaccard_score(answers)
        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt
        return sum([max(0, 1 - lambda_k) for lambda_k in np.linalg.eig(L)[0]])


    def U_DegMat(self, answers):
        #The Degree Matrix
        if self.similarity_score == "NLI_score":
            W = self.compute_adjaency_mat(answers)
        elif self.similarity_score == "Jaccard_score":
            W = self.compute_Jaccard_score(answers)
        D = np.diag(W.sum(axis=1))
        return np.trace(len(answers) - D) / (len(answers) ** 2)

    def U_Eccentricity(self, answers, k = 2):
        if self.similarity_score == "NLI_score":
            W = self.compute_adjaency_mat(answers)
        elif self.similarity_score == "Jaccard_score":
            W = self.compute_Jaccard_score(answers)
        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        smallest_eigenvectors = eigenvectors[:, :k]
        V_mat = smallest_eigenvectors - smallest_eigenvectors.mean(axis = 0)


        norms = np.linalg.norm(V_mat, ord = 2, axis=0)
        U_Ecc = np.linalg.norm(norms, 2)
        C_Ecc_s_j = norms
        return U_Ecc, C_Ecc_s_j
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:

        res = []
        for answers in stats['sample_texts']:
            
            if self.verbose:
                print(f"generated answers: {answers}")
            
            if self.uncertainty_method == 'Num_Sem_Sets':
                print(self.similarity_score)
                assert self.similarity_score == "NLI_score", "For Num_Set_Sets uncertaimty measure only NLI score is relevant"
                res.append(self.U_NumSemSets(answers))

            elif self.uncertainty_method == 'Eig_Val_Laplacian':
                res.append(self.U_EigVal_Laplacian(answers))
            
            elif self.uncertainty_method == 'Deg_Mat':
                res.append(self.U_DegMat(answers))
            
            elif self.uncertainty_method == 'Eccentricity':
                res.append(self.U_Eccentricity(answers)[0])
            
        return np.array(res)